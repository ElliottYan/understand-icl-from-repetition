import io
from setuptools import setup

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='analyze_icl_rep',
    version='0.4',
    license='MIT',
    description='A toolkit for analyzing self-reinforcement effect for In-context Learning',
    author='Jianhao Yan',
    author_email='yanjianhao@westlake.edu.cn',
    url='https://github.com/ElliottYan/understand-icl-from-repetition',
    keywords=['in-context learning', 'self-reinforce', 'nlp'],
    install_requires=[
        "tqdm",
        "scikit-learn",
        "scipy",
        "transformers>=4.30.1",
        "pandas",
        "torch",
        "numpy",
        "setuptools"
    ]
)