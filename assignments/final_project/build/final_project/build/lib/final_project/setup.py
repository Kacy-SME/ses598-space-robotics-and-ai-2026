from setuptools import setup
import os
from glob import glob

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ROS2 ament index marker
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),

        # Terrain mesh model
        (os.path.join('share', package_name, 'models', 'terrain'),
            glob('models/terrain/model.*')),
        (os.path.join('share', package_name, 'models', 'terrain', 'meshes'),
            glob('models/terrain/meshes/*')),

        # PX4 models (airframe + gz model)
        (os.path.join('share', package_name, 'models', 'px4_models', 'airframes'),
            glob('models/px4_models/airframes/*')),
        (os.path.join('share', package_name, 'models', 'px4_models', 'gz_models',
                      'x500_depth_mono'),
            glob('models/px4_models/gz_models/x500_depth_mono/*')),
        (os.path.join('share', package_name, 'models', 'px4_models', 'gz_models',
                      'x500_depth_mono', 'thumbnails'),
            glob('models/px4_models/gz_models/x500_depth_mono/thumbnails/*')),

        # Config
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kacy',
    maintainer_email='kacy@todo.todo',
    description='SES 598 Final Project: Concept-Driven Sparse Landmark Mapping',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'concept_terrain_mission = final_project.concept_terrain_mission:main',
        ],
    },
)
