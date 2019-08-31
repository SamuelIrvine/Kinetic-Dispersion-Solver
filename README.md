# Kinetic-Dispersion-Solver
Solves the kinetic dispersion relation for a homogenous magnetized plasma characterized by arbitrary gyrotropic velocity distribution functions. 

### Installation
These commands work for a fresh Ubuntu 18.04 installation. 

The boost-python-numpy module was only merged in Boost version 1.63.0. In some older linux distributions it may be necessary to manually build this. 

```
sudo apt-get install python-pip jupyter make cmake libboost-all-dev
```

If not already installed, install the following python modules:

```
sudo pip install numpy scipy
```

Check that building works prior to attempting to install. 

```
python setup.py build
```

Now install. 

```
sudo python setup.py install
```

### Getting started

We recommend that users begin by modifying one of the four provided examples [examples](examples). 

## Authors

* **Samuel W A Irvine**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

