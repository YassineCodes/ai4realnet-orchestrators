# AI4REALNET ATM - Test runner for the validation campaign hub

## Table of Contents
- [Overview](#overview)
- [Additional dependencies and installation](#Additional-dependencies-and-installation)
- [Authors](#authors)

## Overview
Scenario files (contained in this repository) and the BlueSky plugin (ai4realnet_deploy_RL_batch.py from the ai4realnet/bluesky repository) can be used as a template by the KPI owners to design/request scenarios.
KPI computation methods can be added to `test_runner.py`, linked to specific scenarios ids (which correspond to specific KPI IDs).

## Additional dependencies and installation
For the infrastructure to work, BlueSky (ai4realnet version) should be installed in editable mode in the environment. The placeholder plugin folder in this repository should be substituted with an alias of the plugin folder of the local installation of bluesky. 

```pip install -e "path/to/local/ai4realnet/bluesky"```

```pip install stable-baselines3```


## Authors
- [Joost Ellerbroek](https://github.com/jooste)
- [Giulia Leto](https://github.com/giulialeto)

Contact Giulia Leto for support.
