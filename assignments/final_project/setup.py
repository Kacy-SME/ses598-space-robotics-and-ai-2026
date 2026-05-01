from setuptools import setup

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/concept_mission_launch.py',
                                               'launch/concept_mission_v3_launch.py',
                                               'launch/concept_mission_v4_launch.py']),
        ('share/' + package_name + '/models/terrain', 
         ['models/terrain/model.sdf', 'models/terrain/model.config']),
        ('share/' + package_name + '/models/terrain/meshes', 
         ['models/terrain/meshes/artburysol175.stl']),
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
            'mission_viz = final_project.mission_viz_node:main',
            'concept_terrain_mission_v3 = final_project.concept_terrain_mission_v3:main',
            'concept_terrain_mission_v4 = final_project.concept_terrain_mission_v4:main',
        ],
    },
)
