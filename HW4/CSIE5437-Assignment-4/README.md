# CSIE5437 Assignment 4 Sample Code

## Installation

[Install OpenFHE in a system directory](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/linux.html)

First clone the [OpenFHE repo](https://github.com/openfheorg/openfhe-development).

Then run the following commands.
```
cd openfhe-development

mkdir build
cd build
cmake ..

make -j4
sudo make install
```

## Download data

Download the `data.zip` file from [here](https://drive.google.com/file/d/1ycj_c1rSHvi3TVtSX_y_pdnno_TeWlfv/view?usp=sharing).

```
cd build
unzip data.zip
```

## Build

Construct Makefile.
```
cd build
cmake ..
```

Compile and generate executable. (under directory `build`)
```
make
```

## Run

Run the executable. (under directory `build`)
```
./main
```

## Contact

Please feel free to contact me through tracyliu981220@gmail.com if you have any questions.
