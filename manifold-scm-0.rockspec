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
}

build = {
   type = "builtin",
   modules = {
       ['manifold.init'] = 'init.lua',
   }
}
