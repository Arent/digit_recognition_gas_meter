from setuptools import setup, find_packages

setup(
    name="stadswarmte_sensor",
    version="0.1.0",
    packages=find_packages(include=["meterkast_data", "meterkast_data.*"]),
    install_requires=[] # Just install your packages yourself
        
    entry_points={
        "console_scripts": [
            "run_meterkast_sensor=stadswarmte_sensor.main:main",
            "just_take_images=stadswarmte_sensor.just_take_images:main",
        ]
    },
)
