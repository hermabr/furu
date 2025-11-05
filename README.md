# huldra

NOTE: this is a very early prototype, expect breaking changes. for now this is mostly an idea

huldra makes it trivial to create minimal, cacheable pipelines in Python by extending dataclasses.

all huldra objects map from a config to artifact, using nested config objects for defining the pipelines

## how to use

1. inherit from Huldra
2. define the functions `_create` for creating an object and `_load` for loading existing object
3. call the method `load_or_create` to load the object if it already exists or create if it does not already exist

## features

- [ ] stores metadata for which code generated your results and how to revert to them
- [ ] stores state for project while it is running
- [ ] integrates with submitit
