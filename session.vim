let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/projects/hms
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +47 dataset.py
badd +0 term://.//4169:/bin/bash
badd +0 model.py
argglobal
silent! argdel *
$argadd dataset.py
edit model.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd t
set winminheight=1 winminwidth=1 winheight=1 winwidth=1
exe '1resize ' . ((&lines * 46 + 34) / 68)
exe 'vert 1resize ' . ((&columns * 112 + 114) / 229)
exe '2resize ' . ((&lines * 46 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 116 + 114) / 229)
exe '3resize ' . ((&lines * 19 + 34) / 68)
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 31 - ((30 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
31
normal! 010|
wincmd w
argglobal
if bufexists('dataset.py') | buffer dataset.py | else | edit dataset.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 80 - ((41 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
80
normal! 04|
lcd ~/projects/hms
wincmd w
argglobal
if bufexists('term://.//4169:/bin/bash') | buffer term://.//4169:/bin/bash | else | edit term://.//4169:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 88 - ((7 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
88
normal! 0103|
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 46 + 34) / 68)
exe 'vert 1resize ' . ((&columns * 112 + 114) / 229)
exe '2resize ' . ((&lines * 46 + 34) / 68)
exe 'vert 2resize ' . ((&columns * 116 + 114) / 229)
exe '3resize ' . ((&lines * 19 + 34) / 68)
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToO
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
