from setuptools import setup, find_packages

setup(
    name='finger_optimizer',
    version='0.0.0',
    packages=find_packages(),  # Will automatically find 'finger_optimizer' package
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/finger_optimizer']),
        ('share/finger_optimizer', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
)