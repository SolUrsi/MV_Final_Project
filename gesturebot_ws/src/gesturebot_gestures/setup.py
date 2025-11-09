from setuptools import find_packages, setup

package_name = 'gesturebot_gestures'

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
    maintainer='fynns',
    maintainer_email='fynns@uia.no',
    description='A package to detect hand gestures used to control a robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_detector = gesturebot_gestures.gesture_detector:main',
        ],
    },
)
