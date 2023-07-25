from setuptools import setup
from setuptools import find_packages


VERSION = "0.2.1"

with open("README.en.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="ai4xde",  # package name
    version=VERSION,  # package version
    description="AI4XDE is a library for scientific machine learning and physics-informed learning",  # package description
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="xuelanghanbao",  # author name
    author_email="xuelanghanbao@gmail.com",
    project_urls={
        "Code": "https://gitee.com/xuelanghanbao/AI4XDE",
        "Issue tracker": "https://gitee.com/xuelanghanbao/AI4XDE/issues",
    },
    license="GNU Lesser General Public License v2 (LGPLv2) (LGPL-2.1)",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "deepxde",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    zip_safe=True,
)
