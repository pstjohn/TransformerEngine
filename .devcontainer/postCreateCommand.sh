#!/bin/bash
ln -s /usr/local/lib/python3.12/dist-packages/transformer_engine/*.so /workspaces/TransformerEngine/transformer_engine/
export PYTHONPATH=/workspaces/TransformerEngine:$PYTHONPATH
