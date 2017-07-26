from fabric.api import local, lcd, env, cd, run
from fabric.contrib.project import rsync_project
import subprocess
import os

env.use_ssh_config = True
env.hosts = [
    # 'iis-dev',
   'megacomplete-aws',
]

def deploy():
    local('git push')
    git_rev = subprocess.check_output(['git', 'describe', '--always']).decode('utf-8').strip()
    open('frontend/.env', 'w').write(f'REACT_APP_GIT_REV={git_rev}\n')
    with cd('~/code/suggestion'):
        run('git pull')
    with lcd('frontend'):
        local('npm run build')
    rsync_project(remote_dir='~/code/suggestion/frontend/build/', local_dir='frontend/build/', delete=True)
    # rsync -Pax models/ megacomplete-aws:/home/ubuntu/code/suggestion/models/
    rsync_project(remote_dir='~/code/suggestion/models/', local_dir='models/', delete=True)
    with lcd('frontend'):
        local(f'sentry-cli releases -o kenneth-arnold -p suggestionfrontend new {git_rev}')
        local(f'sentry-cli releases -o kenneth-arnold -p suggestionfrontend files {git_rev} upload-sourcemaps src build')

def get_data():
    subprocess.run(['./pull-logs'], env=dict(os.environ, SERVER='megacomplete-aws'))
    subprocess.run(['python', 'scripts/get_surveys.py'])
