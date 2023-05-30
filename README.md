# Population Analysis and Animation

<div align="center">
<img src="https://i.imgur.com/he6g0UR.png" width="68px" height="68px"/>
</div>

![MIT License](https://img.shields.io/github/license/devops-master-4/hadoop)
![GitHub Repo stars](https://img.shields.io/github/stars/devops-master-4/hadoop?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/devops-master-4/hadoop?style=social)
![Python](https://img.shields.io/badge/python-3.8-blue)
![PySpark](https://img.shields.io/badge/pyspark-3.1.2-blue)
![matplotlib](https://img.shields.io/badge/matplotlib-3.4.2-blue)
![sklearn](https://img.shields.io/badge/sklearn-0.24.2-blue)

This project performs analysis on population data using PySpark and creates an animated visualization of the population by year in the world using matplotlib.

`scikit-learn` is used to perform a linear regression on the data to predict the population in 2030.

## Table of Contents

- [Population Analysis and Animation](#population-analysis-and-animation)
- [Table of Contents](#table-of-contents)
- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

![Population Animation](https://i.imgur.com/0qm1Np5.gif)

## Requirements

- Python 3.x
- PySpark
- matplotlib
- sklearn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/devops-master-4/hadoop.git
cd hadoop
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

You can also format the data with docker:

```bash
# go to the hadoop-spark2023 folder
cd hadoop-spark2023/
# build the docker image
docker build -t spark2023 .
# run the docker image
docker compose -f "docker-compose.yml" up -d --build
# you can find the formatted data in the data folder in the docker container
cd /home/hadoop/data && data/population_formatted.csv
```

## Usage

1. Run the script:

```bash
python3 main.py
```

2. The script will create a folder called `output` and save the animation as `population.gif` in it.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

> _Note: This is only for educational purposes. The data used in this project may not be accurate, and the results may not be correct. The data is taken from [Kaggle](https://www.kaggle.com/)._

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

- @luc-lecoutour
- @meteor314
