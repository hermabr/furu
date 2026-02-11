# Plan for a better implementation of the executor

There are three modes:

- Local executor
- Slurm worker pool
- Slurm dag (this one is already implemented in a reasonable way)

## Local executor

The main API will remain as today

```
run_local(
    [TrainModel(lr=3e-4, steps=5000), TrainModel(lr=1e-3, steps=2000)],
    max_workers=8,
    window_size="bfs",
)
```

When the worker is first submitted, the following should happen:

It builds a state with all the tasks that need to be completed. It will have the following states. We will only ever do this once

- TODO (includes the hashes of everything that depends on it and everything that depends on it)
- Failed
- InProgress
- ExternalInProgress (the jobs currently running, but not through our executor)
- Completed

After the initial plan is made

- Whenever an InProgress is completed or an ExternalInProgress is completed, do the following. This should not happen on a timer
    - If InProgress is less than allowed number of concurrent jobs, start new ones from TODO if there are jobs in TODO that are ready to be executed (no pending dependencies)
        - Before submitting any job from TODO, make sure its not already completed or running in an ExternalInProgress. If it is, simply move that node to the correct bucket and try submitting another job.
            - If it is completed, remove it as a dependency in all the TODOs that have it as a dependency (the object itself should have a list of these tasks)
- If there are fewer InProgress items than the max concurrent limit (e.g., every 15 seconds: user configurable)
    - It should go through all the TODO and Failed objects and check if any of them have changed state by moving it to the correct bucket and change state of other nodes, such as if a node is now completed
- If there are any items in ExternalInProgress, it should poll every so often (e.g., every 5 seconds: user configurable)
    - Check if any of them have changed state and if they do, handle that by moving it to the correct bucket and change state of other nodes, such as if a node is now completed
