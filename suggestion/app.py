import re
import json
import time
import os
import traceback
import datetime
import io
import zlib

import tornado.ioloop
import tornado.gen
import tornado.options
import tornado.web
import tornado.websocket

from .paths import paths

import subprocess

def get_git_commit():
    return subprocess.check_output(['git', 'describe', '--always']).decode('ascii').strip()

from tornado.options import define, options

define("port", default=5000, help="run on the given port", type=int)

settings = dict(
    template_path=paths.ui,
    static_path=paths.ui / 'static',
    debug=True,
    )

server_settings = dict(
    address='127.0.0.1',
    xheaders=True)

from . import suggestion_generator

# Convert the normal generator function into a Tornado coroutine.
# We do this here to avoid tornado imports in the core suggestion_generator.
get_suggestions_async = tornado.gen.coroutine(suggestion_generator.get_suggestions_async)

if not os.path.isdir(paths.logdir):
    os.makedirs(paths.logdir)

from concurrent.futures import ProcessPoolExecutor
process_pool = ProcessPoolExecutor()

import tornado.autoreload
tornado.autoreload.add_reload_hook(process_pool.shutdown)


known_participants = {}


class Participant:
    @classmethod
    def get_participant(cls, participant_id):
        if participant_id in known_participants:
            return known_participants[participant_id]
        participant = cls(participant_id)
        known_participants[participant_id] = participant
        return participant

    def __init__(self, participant_id):
        self.participant_id = participant_id
        self.connections = []

        self.log_file_name = os.path.join(paths.logdir, self.participant_id + '.jsonl')
        self.log_file = open(self.log_file_name, 'a+')
        self.log_file.seek(0, io.SEEK_END)

    def log(self, event):
        assert self.log_file is not None
        print(
            json.dumps(dict(event, timestamp=datetime.datetime.now().isoformat(), participant_id=self.participant_id)),
            file=self.log_file, flush=True)

    def get_log_entries(self):
        self.log_file.seek(0)
        log_entries = [json.loads(line) for line in self.log_file]
        self.log_file.seek(0, io.SEEK_END)
        return log_entries

    def broadcast(self, msg, exclude_conn):
        for conn in self.connections:
            if conn is not exclude_conn:
                conn.send_json(**msg)
        Panopticon.spy(msg)

    def connected(self, client):
        self.connections.append(client)
        print(client.kind, 'open', self.participant_id)
        self.log(dict(kind='meta', type='connected', rev=get_git_commit()))

    def disconnected(self, client):
        self.connections.remove(client)
        print(client.kind, 'close', self.participant_id)


class DemoParticipant:
    participant_id = 'DEMO'

    def get_log_entries(self):
        return []

    def log(self, event): return
    def broadcast(self, *a, **kw): return
    def connected(self, *a, **kw): return
    def disconnected(self, *a, **kw): return

class Panopticon:
    is_panopticon = True
    participant_id = 'panopt'
    connections = []

    @classmethod
    def spy(cls, msg):
        for conn in cls.connections:
            conn.send_json(**msg)

    def get_log_entries(self):
        entries = []
        for participant in known_participants.values():
            entries.extend(participant.get_log_entries())
        return entries

    def log(self, event): return
    def broadcast(self, *a, **kw): return
    def connected(self, client):
        Panopticon.connections.append(client)
    def disconnected(self, client):
        Panopticon.connections.remove(client)


def validate_participant_id(participant_id):
    return re.match(r'^[0-9a-zA-Z]+$', participant_id) is not None


class WebsocketHandler(tornado.websocket.WebSocketHandler):
    def get_compression_options(self):
        # Non-None enables compression with default options.
        return None

    def on_close(self):
        if self.participant is not None:
            self.participant.disconnected(self)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.participant = None
        self.keyRects = {}
        self.wire_bytes_in = self.wire_bytes_out = 0
        self.message_bytes_in = self.msg_bytes_out = 0
        self.sug_state = None
        self.connection_id = str(time.time())
        # There will also be a 'kind', which gets set only when the client connects.

    def log(self, event):
        self.participant.log(dict(event))

    def open(self):
        print('ws open, compressed={}'.format(self.ws_connection._compressor is not None), flush=True)
        self.inflater = zlib.decompressobj()
        self.deflater = zlib.compressobj()

    def send_json(self, **kw):
        message = json.dumps(kw)
        self.msg_bytes_out += len(message)
        message = self.deflater.compress(message.encode('utf-8'))
        message += self.deflater.flush(zlib.Z_SYNC_FLUSH)
        self.wire_bytes_out += len(message)
        self.write_message(message.decode('latin1'))

    @tornado.gen.coroutine
    def on_message(self, message):
        self.wire_bytes_in += len(message)
        message = self.inflater.decompress(message.encode('latin1'))
        message += self.inflater.flush()
        message = message.decode('utf-8')
        self.message_bytes_in += len(message)
        try:
            request = json.loads(message)
            if request['type'] == 'requestSuggestions':
                start = time.time()
                request_id = request.get('request_id')
                # Clear the suggestion state when starting a new experiment.
                if request_id == 0:
                    self.sug_state = None
                result = dict(type='suggestions', timestamp=request['timestamp'], request_id=request_id)
                flags = request['flags']
                if flags.get('split'):
                    result.update(suggestion_generator.get_split_recs(request['sofar'], request['cur_word'], request['flags']))
                elif flags.get('alternatives'):
                    result.update(suggestion_generator.get_clustered_recs(request['sofar'], request['cur_word'], request['flags']))
                else:
                    suggestion_kwargs = suggestion_generator.request_to_kwargs(flags)
                    try:
                        phrases, self.sug_state = yield get_suggestions_async(
                            process_pool,
                            sofar=request['sofar'], cur_word=request['cur_word'],
                            sug_state=self.sug_state,
                            **suggestion_kwargs)
                    except Exception:
                        traceback.print_exc()
                        print("Failing request:", json.dumps(request))
                        phrases = []
                    result['predictions'] = suggestion_generator.phrases_to_suggs(phrases)
                dur = time.time() - start
                result['dur'] = dur
                self.send_json(**result)
                self.log(dict(type="requestSuggestions", kind="meta", request=request))
                print('{participant_id} {type} in {dur:.2f}'.format(participant_id=getattr(self.participant, 'participant_id'), type=request['type'], dur=dur))
            elif request['type'] == 'keyRects':
                self.keyRects[request['layer']] = request['keyRects']
            elif request['type'] == 'init':
                participant_id = request['participantId']
                self.kind = request['kind']
                if participant_id.startswith('demo'):
                    self.participant = DemoParticipant()
                elif self.kind == 'panopt' and participant_id == '42':
                    self.participant = Panopticon()
                else:
                    validate_participant_id(participant_id)
                    self.participant = Participant.get_participant(participant_id)
                self.participant.connected(self)
                self.log(dict(kind='meta', type='init', request=request))
                messageCount = request.get('messageCount', {})
                print("Client", participant_id, self.kind, "connecting with messages", messageCount)
                backlog = []
                cur_msg_idx = {}
                for entry in self.participant.get_log_entries():
                    kind = entry['kind']
                    if kind == 'meta':
                        continue
                    idx = cur_msg_idx.get(kind, 0)
                    if idx >= messageCount.get(kind, 0):
                        backlog.append(entry)
                    cur_msg_idx[kind] = idx + 1
                self.send_json(type='backlog', body=backlog)

            elif request['type'] == 'get_logs':
                assert self.participant.is_panopticon
                participant_id = request['participantId']
                validate_participant_id(participant_id)
                participant = Participant.get_participant(participant_id)
                self.send_json(type='logs', participant_id=participant_id, logs=participant.get_log_entries())

            elif request['type'] == 'get_analyzed':
                assert self.participant.is_panopticon
                participant_id = request['participantId']
                validate_participant_id(participant_id)
                from .analysis_util import get_log_analysis
                # TODO: include the git revision overrides here
                analysis = get_log_analysis(participant_id)
                self.send_json(type='analyzed', participant_id=participant_id, analysis=analysis)

            elif request['type'] == 'log':
                event = request['event']
                self.log(event)
                self.participant.broadcast(dict(type='otherEvent', event=event), exclude_conn=self)
            elif request['type'] == 'ping':
                pass
            else:
                print("Unknown request type:", request['type'])
            # print(', '.join('{}={}'.format(name, getattr(self.ws_connection, '_'+name)) for name in 'message_bytes_in message_bytes_out wire_bytes_in wire_bytes_out'.split()))
            # print('wire i={wire_bytes_in} o={wire_bytes_out}, msg i={message_bytes_in} o={msg_bytes_out}'.format(**self.__dict__))
        except Exception:
            traceback.print_exc()

    def check_origin(self, origin):
        """Allow any CORS access."""
        return True


class WSPingHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('pinger open, compressed={}'.format(self.ws_connection._compressor is not None), flush=True)

    def on_message(self, message):
        self.write_message(message)

    def check_origin(self, origin):
        """Allow any CORS access."""
        return True


class MainHandler(tornado.web.RequestHandler):
    def head(self):
        self.finish()

    def get(self):
        self.render("index.html")


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', MainHandler),
            (r"/ws", WebsocketHandler),
            (r'/ping', WSPingHandler),
            (r"/(style\.css)", tornado.web.StaticFileHandler, dict(path=paths.ui)),
        ]
        tornado.web.Application.__init__(self, handlers, **settings)


def main():
    tornado.options.parse_command_line()
    app = Application()
    print('serving on', options.port)
    app.listen(options.port, **server_settings)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
