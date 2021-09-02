# mcts-demo
A basic implementation of the Monte Carlo Tree Search algorithm, with concrete illustrations based on some toy examples. Made for educational purposes.

<p align="center">
    <img src="https://github.com/lucasBOYER/mcts-demo/img/pendulum.gif" width="300" height="300" alt="MCTS-agent playing Pendulum"/>
    <em>MCTS-agent playing Pendulum-v0</em>
</p>

# Installation
## With docker
While being at the root of the project, build the image with the following command :
```sh
docker build -t mcts-demo
```

Then run the container that will run the JupyterLab instance with:
```sh
docker run -p 8888:8888 -v <absolute/path/to/project/root>:/home/jovyan/work/ -e JUPYTER_ENABLE_LAB=yes mcts-demo
```
* `-v <absolute/path/to/project/root>:/home/jovyan/work/` makes all the root folder on the host linked to the `work` folder (the default working directory) on the container
* `-p 8888:8888` links the port 8888 on the container (default port that JupyterLab will use) to the port 8888 of the host
* `-e JUPYTER_ENABLE_LAB=yes` will activate JupyterLab (instead of a classic Jupyter Notebook server)

More arguments to control JupyterLab's behavior [here](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html).

## With poetry

Install [poetry] and use it to manage the project's dependencies:
```sh
pip install poetry
poetry install
```

Poetry will by default create a virtual environment to isolate the project from the main python installation (see the official [poetry documentation](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) for more information on that topic).

One can check that the installation went properly by running the tests from the poetry environment with `poetry run`
```sh
poetry run pytest ./tests
```

To play with the notebooks, one can then run a JupyterLab instance from withing a `poetry shell` to activate the relevant kernel:
```sh
poetry shell
```
```sh
jupyter lab
```




[poetry]: https://python-poetry.org/docs/

