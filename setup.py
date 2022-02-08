from setuptools import setup, find_packages

setup(
    name='task_strategy_learning',
    version='1.0.0',
    description='Extracting behaviors of an RL agent via contrastive representation learning.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'scikit-image',
        'pillow==6.2.1',
        'yacs',
        'tqdm',
        'tensorboard',
        'torchsummary',
        'scikit-learn==0.24.2',
    ]
)