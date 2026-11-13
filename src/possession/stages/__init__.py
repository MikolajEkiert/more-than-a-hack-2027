"""One module per stage. Importing this package registers every method.

To add a method: write a registered function in the module of its stage (or a new module in this
package, imported below). Never add a second pipeline.
"""

from possession.stages import ball_state, decode, team  # noqa: F401
