import os
import ssl

def hack():
  
    # --------------------------------------------------------------------
    # disable the error:
    # `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA`
    # --------------------------------------------------------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --------------------------------------------------------------------
    # hack to avoid SSL download error
    # --------------------------------------------------------------------
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context


