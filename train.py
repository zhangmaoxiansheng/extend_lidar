# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function


from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()
if opts.refine and not opts.dropout:
    from trainer_cspnall import Trainer
elif opts.dropout:
    from trainer_selftest import Trainer
else:
    from trainer_dep import Trainer


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
