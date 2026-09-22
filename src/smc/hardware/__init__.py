"""Hardware access.

Everything that touches a device lives under this package (ADR-0002/0003).
Plugins and user interfaces never import a vendor SDK or a Micro-Manager
device name directly: they ask for a *capability* by *role*, and this
package works out which device fills it.
"""
