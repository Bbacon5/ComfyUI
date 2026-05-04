import functools
import logging
import os

import folder_paths

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOY_ENV = "local_git"
_ENV_FILENAME = ".comfy_environment"


@functools.cache
def get_deploy_environment() -> str:
    env_file = os.path.join(folder_paths.base_path, _ENV_FILENAME)
    try:
        with open(env_file, encoding="utf-8") as f:
            # Cap the read so a malformed or maliciously crafted file (e.g.
            # a single huge line with no newline) can't blow up memory.
            first_line = f.readline(128).strip()
            value = "".join(c for c in first_line if 32 <= ord(c) < 127)
            if value:
                return value
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("Failed to read %s: %s", env_file, e)

    return _DEFAULT_DEPLOY_ENV
