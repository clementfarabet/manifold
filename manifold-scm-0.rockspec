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
   type = "builtin",
   modules = {
       ['manifold.init'] = 'init.lua',
       ['manifold.tsne'] = 'tsne.lua',
       ['manifold.lle'] = 'lle.lua',
       ['manifold.laplacian_eigenmaps'] = 'laplacian_eigenmaps.lua',
   },
   install = {
       bin = {
           'bhtsne_maci',
           'bhtsne_linux',
       }
   }
}
