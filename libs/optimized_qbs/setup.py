from distutils.core import setup, Extension

module = Extension('cqbs', sources = ['qbs.c', 'cqbsmodule.c'])

setup (name = 'cqbs',
	   version = '1.0',
	   description = 'C implementation of QBSes',
	   ext_modules = [module])