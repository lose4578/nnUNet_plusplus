def Settings( **kwargs ):
  return {
    'sys_path': [
        '/home/lidouzhe/Documents/nnUNet/nnunet/'
    ]
  }

def PythonSysPath( **kwargs ):
  sys_path = kwargs[ 'sys_path' ]
  sys_path.insert( 1, '~/Documents/nnUNet/nnunet/')
  return sys_path
