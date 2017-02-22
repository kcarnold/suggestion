import string
import random
import json
import time
import os
import traceback
import datetime
import io

import tornado.ioloop
import tornado.gen
import tornado.options
import tornado.web
import tornado.websocket

from .paths import paths


from tornado.options import define, options

define("port", default=5000, help="run on the given port", type=int)

settings = dict(
    xheaders=True,
    template_path=paths.ui,
    static_path=paths.ui,
    debug=True,
    )

from . import suggestion_generator

if not os.path.isdir(paths.logdir):
    os.makedirs(paths.logdir)

import sqlite3
db_conn = sqlite3.connect(paths.db)
db_conn.execute('PRAGMA journal_mode=WAL;')
db_conn.execute('PRAGMA synchronous=NORMAL;')
db_conn.execute('CREATE TABLE IF NOT EXISTS sessions (participant_id, log_file, state)')

from concurrent.futures import ProcessPoolExecutor
process_pool = ProcessPoolExecutor()

import tornado.autoreload
tornado.autoreload.add_reload_hook(process_pool.shutdown)


INITIAL_STATE = dict()

active_participants = {}


class Participant:
    @classmethod
    def get_participant(cls, participant_id):
        if participant_id in active_participants:
            return active_participants[participant_id]
        participant = cls(participant_id)
        active_participants[participant_id] = participant
        return participant

    def __init__(self, participant_id):
        self.participant_id = participant_id
        self.connections = []

        self.log_file_name = os.path.join(paths.logdir, self.participant_id + '.jsonl')
        self.log_file = open(self.log_file_name, 'a+')
        self.log_file.seek(0, io.SEEK_END)
        if self.state is None:
            self.state = dict(INITIAL_STATE)

    def log(self, event):
        assert self.log_file is not None
        print(
            json.dumps(dict(event, timestamp=datetime.datetime.now().isoformat(), participant_id=self.participant_id)),
            file=self.log_file, flush=True)

    @property
    def state(self):
        with db_conn:
            state_json = db_conn.execute('SELECT state FROM sessions WHERE participant_id=?', (self.participant_id,)).fetchone()
        if state_json is None:
            return None
        else:
            return json.loads(state_json[0])

    @state.setter
    def state(self, state):
        self.log(dict(set_state=state))
        prev_state = self.state
        if prev_state is not None:
            state = dict(prev_state, **state)
            with db_conn:
                db_conn.execute('UPDATE sessions SET state=? WHERE participant_id=?', (
                    json.dumps(state), self.participant_id))
        else:
            with db_conn:
                db_conn.execute('INSERT INTO sessions VALUES (?, ?, ?)', (
                    self.participant_id, self.log_file_name, json.dumps(state)))

        self.send_to_clients(state=state)
        self.send_to_controllers(state=state)

    def get_log_entries(self):
        self.log_file.seek(0)
        log_entries = [json.loads(line) for line in self.log_file]
        self.log_file.seek(0, io.SEEK_END)
        return log_entries


    def send_to_clients(self, **kw):
        for conn in self.connections:
            if conn.kind == 'client':
                conn.send_json(**kw)

    def send_to_controllers(self, **kw):
        for conn in self.connections:
            if conn.kind == 'controller':
                conn.send_json(**kw)

    def connected(self, client):
        self.connections.append(client)
        client.send_json(state=self.state)
        print(client.kind, 'open', self.participant_id)

    def disconnected(self, client):
        self.connections.remove(client)
        print(client.kind, 'close', self.participant_id)


class MyWSHandler(tornado.websocket.WebSocketHandler):
    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def send_json(self, **kw):
        self.write_message(json.dumps(kw))

    def on_close(self):
        if self.participant is not None:
            self.participant.disconnected(self)


class WebsocketHandler(MyWSHandler):
    kind = 'client'
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log_file = None
        self.participant = None
        self.keyRects = {}

    def open(self):
        print('open', flush=True)

    @tornado.gen.coroutine
    def on_message(self, message):
        try:
            start = time.time()
            request = json.loads(message)
            # if self.participant is not None:
            #     self.participant.log(dict(type='clientMessage', msg=request))
            if request['type'] == 'requestSuggestions':
                phrases = yield process_pool.submit(suggestion_generator.get_suggestions,
                    request['sofar'], request['cur_word'], domain=request.get('domain', 'yelp_train'),
                    rare_word_bonus=request.get('rare_word_bonus', 1.0))
                result = dict(type='suggestions', timestamp=request['timestamp'], request_id=request.get('request_id'))
                result['next_word'] = suggestion_generator.phrases_to_suggs(phrases)
                self.send_json(**result)
                print('{type} in {dur:.2f}'.format(type=request['type'], dur=time.time() - start))
            elif request['type'] == 'keyRects':
                self.keyRects[request['layer']] = request['keyRects']
            elif request['type'] == 'setState':
                self.participant.state = request['state']
            elif request['type'] == 'requestBacklog':
                self.send_json(type='backlog', body=self.participant.get_log_entries())
            elif request['type'] == 'init':
                # self.client_id = request['client_id']
                participant_id = request['participantId']
                assert all(x in string.hexdigits for x in participant_id)
                self.participant = Participant.get_participant(participant_id)
                self.participant.connected(self)
            elif request['type'] == 'log':
                self.participant.log(request['event'])
                self.participant.send_to_controllers(client_log=request['event'])
            elif request['type'] == 'ping':
                pass
            else:
                print("Unknown request type:", request['type'])
        except Exception:
            traceback.print_exc()

    def check_origin(self, origin):
        return True


class ControllerHandler(MyWSHandler):
    kind = 'controller'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.participant = None

    def open(self):
        print('controller open', flush=True)

    def on_message(self, message):
        request = json.loads(message)
        if request['type'] == 'init':
            participant_id = request['participantId']
            assert all(x in string.hexdigits for x in participant_id)
            self.participant = Participant.get_participant(participant_id)
            self.participant.connected(self)
        elif request['type'] == 'setState':
            self.participant.state = request['newState']


class Admin(MyWSHandler):
    def on_message(self, message):
        request = json.loads(message)
        if request['type'] == 'dump':
            res = []
            for participant_id, participant in active_participants.items():
                res.append(dict(
                    participant_id=participant_id,
                    state=participant.state,
                    active=len(participant.client_connections) or len(participant.controller_connections)))
            self.send_json(dump=res)
        elif request['type'] == 'setState':
            active_participants[request['participant_id']].state = request['state']
        elif request['type'] == 'sendMsg':
            participant_id = request['participant_id']
            msg = request['msg']
            participant = active_participants[participant_id]
            if request.get('controller', True):
                participant.send_to_controllers(**msg)
            if request.get('client', True):
                participant.send_to_clients(**msg)

    def check_origin(self, origin):
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
            (r"/wsController", ControllerHandler),
            (r"/wsAdmin", Admin),
        ]
        tornado.web.Application.__init__(self, handlers, **settings)


def main():
    tornado.options.parse_command_line()
    app = Application()
    print('serving on', options.port)
    app.listen(options.port, address='127.0.0.1')
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
