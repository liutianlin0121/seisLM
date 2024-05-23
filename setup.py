import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='seisLM',
    version='0.1',
    author='Tianlin Liu',
    author_email='t.liu@unibas.ch',
    description='Seisbench language model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/liutianlin0121/seisbench-lm',
    install_requires=reqs,
    packages=setuptools.find_packages()
)

