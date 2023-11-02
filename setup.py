import setuptools
 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scarp", 
    version="1.0.0",    
    author="Jiating Yu",    
    author_email="yujiating@amss.ac.cn",    
    description="Incorporating network diffusion and peak location information for better single-cell ATAC-seq data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wu-lab/SCARP",
    install_requires=[
        'scanpy >= 1.9.5',
        'pandas >= 2.1.1',
        'numpy >= 1.25.2',
        'scipy >= 1.11.3'
    ],
    packages=['scarp'],
    python_requires='>=3.10',
)