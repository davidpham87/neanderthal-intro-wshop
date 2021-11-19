# Requirements

In order to insure a smooth experience,

- Docker
- Babashka

If you do not installed intel-mkl on your pc, you would need Docker and bashaka (as bb) on your path.

``` shell
bb docker-build
bb docker-run
bb docker-repl # binds a repl to 8889, a portal to 53755 and clerk to 7777
```
