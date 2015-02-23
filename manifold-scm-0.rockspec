package = "manifold"
version = "scm-0"

source = {
   url = "git://github.com/clementfarabet/manifold"
}

description = {
   summary = "A package to manipulate manifolds",
   detailed = [[
A package to manipulate manifolds.
   ]],
   homepage = "https://github.com/clementfarabet/manifold",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "unsup",
   "mnist",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install",
}
