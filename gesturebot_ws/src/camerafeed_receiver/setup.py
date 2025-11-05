from setuptools import find_packages, setup

package_name = 'camerafeed_receiver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bjørn Sødal',
    maintainer_email='bjornso@uia.no',
    description='ROS 2 package to subscribe to a compressed camera feed and display it using OpenCV.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feed_receiver = camerafeed_receiver.feed:main',
        ],
    },
)
