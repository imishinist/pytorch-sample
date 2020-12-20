# PyTorch example


## dependencies

- python 3.8

## setup venv

```shell script
python3 -m venv .venv
. .venv/bin/activate
```

## install package

```shell script
make install-package
```

## Run

```shell script
python main.py
```

## update package

when you add some package, update requirements.txt.

```shell script
make save-package
```

## Download LibTorch

Download libtorch from [pytorch.org](https://pytorch.org/)
and move to `thircparty/libtorch/` like below.

```
$ tree thirdparty/
thirdparty/
└── libtorch
    ├── bin
    └── include
        └── torch
            └── csrc
                └── api
                    └── include
                        └── torch
                            └── data
                                ├── dataloader
                                │   ├── base.h
                                │   ├── stateful.h
                                │   └── stateless.h
                                ├── dataloader.h
                                ├── dataloader_options.h
                                ├── datasets
                                │   ├── base.h
                                │   ├── chunk.h
                                │   ├── map.h
                                │   ├── mnist.h
                                │   ├── shared.h
                                │   ├── stateful.h
                                │   └── tensor.h
                                ├── datasets.h
                                ├── detail
                                │   ├── data_shuttle.h
                                │   ├── queue.h
                                │   └── sequencers.h
                                ├── example.h
                                ├── iterator.h
                                ├── samplers
                                │   ├── base.h
                                │   ├── custom_batch_request.h
                                │   ├── distributed.h
                                │   ├── random.h
                                │   ├── sequential.h
                                │   ├── serialize.h
                                │   └── stream.h
                                ├── samplers.h
                                ├── transforms
                                │   ├── base.h
                                │   ├── collate.h
                                │   ├── lambda.h
                                │   ├── stack.h
                                │   └── tensor.h
                                ├── transforms.h
                                └── worker_exception.h
```
