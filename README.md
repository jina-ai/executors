> ⚠️ Please do **not** commit your new Executor to this repository. This repository is **only** for Jina engineers to better manage in-house executors in a centralized way. You *may* submit PRs to fix bugs/add features to the existing ones.


# Executors (internal-only)

To develop your own Executor, please use [`jina hub new`](#create-new) and create your own Executor repo.

Here is the complete guide: https://docs.jina.ai/advanced/hub/

## Remark to Internal Developers

### Using this repository as a package

Notice that we have a `setup.py` in this repository. 
This is **NOT recommended** practice for **external developers** of Executors. 
We added this in order to ease local development for **internal developers**.

This file, along with the `__init__.py`s in each of the folders, do not matter when using the Executors via the `jinahub://` syntax, [above](#jinahub).

### CompoundExecutors

If you want to develop a `CompoundExecutor`-type Executor based on one of the classes provided in this package, you can either:

- fork this repo and add it as a separate folder. Start with `jina hub new`, and then follow the design patterns we have established in [here](jinahub/indexers/searcher/compound) and in the [docs](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md).
- copy-paste the class you want to have a component of your `CompoundExecutor`, and add it as a class in your Executor's package

