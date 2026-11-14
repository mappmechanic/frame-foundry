"""Select media executables without shell parsing; otherwise honor PATH."""
import os

def media_binary(name):
    return os.environ.get('FRAMEFOUNDRY_'+name.upper()) or name
