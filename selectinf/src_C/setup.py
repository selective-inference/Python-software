from distutils.core import setup, Extension
from Cython.Build import cythonize

from Cython.Distutils import build_ext
import numpy as np                        



setup(
    #name = 'kmean',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ## ext_modules = cythonize("sample_preparation.pyx", 
    ##                        language="c++")
    ext_modules = [Extension('sampler', 
                             ["sample_preparation.pyx" ,
                             'preparation_Eig_Vect.cpp' ,
                             'HmcSampler.cpp'],
                             language="c++",
                             extra_compile_args = ["-W", 
                                                   "-Wall", 
                                                   "-ansi", 
                                                   "-pedantic", 
                                                   "-stdlib=libstdc++"#, 
                                                   #"-fPIC"
                                               ],
                             extra_link_args = ["-stdlib=libstdc++"]
                             )]
                         
)






