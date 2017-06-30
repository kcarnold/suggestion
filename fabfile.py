from fabric.api import local, lcd, env, cd, run
from fabric.contrib.project import rsync_project
import subprocess

env.use_ssh_config = True
env.hosts = [
    # 'iis-dev',
   'megacomplete-aws',
]

def deploy():
    local('git push')
    git_rev = subprocess.check_output(['git', 'describe', '--always'])
    with cd('~/code/suggestion'):
        run('git pull')
    with lcd('frontend'):
        open('.env', 'w').write(f'GIT_REV={git_rev}')
        local('npm run build')
        local(f'sentry-cli releases -o kenneth-arnold -p suggestionfrontend new {git_rev}')
        local(f'sentry-cli releases -o kenneth-arnold -p suggestionfrontend files {git_rev} upload-sourcemaps src build')
    rsync_project(remote_dir='~/code/suggestion/frontend/build/', local_dir='frontend/build/', delete=True)
