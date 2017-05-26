from fabric.api import local, lcd, env, cd, run
from fabric.contrib.project import rsync_project

env.use_ssh_config = True
env.hosts = [
    'iis-dev',
#    'megacomplete-aws',
]

def deploy():
    local('git push')
    with cd('~/code/suggestion'):
        run('git pull')
    with lcd('frontend'):
        local('npm run build')
    rsync_project(remote_dir='~/code/suggestion/frontend/build/', local_dir='frontend/build/', delete=True)
