import setuptools

setuptools.setup(
    name='jinahub-text-paddle',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executors that encode text with PaddlePaddle',
    url='https://github.com/jina-ai/executor-text-paddle',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.text_paddle'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)