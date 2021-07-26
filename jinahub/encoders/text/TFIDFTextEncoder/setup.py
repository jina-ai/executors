import setuptools

setuptools.setup(
    name='jinahub-tfidf',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executor that encodes text with tfidf',
    url='https://github.com/jina-ai/executor-text-tfidfencoder',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.tfidf_text_executor'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

