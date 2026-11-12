from .report import run
import sys
run(*sys.argv[1:2] or ["config.toml"])
