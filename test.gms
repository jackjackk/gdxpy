set set_1dim / a*c /;
set set_2dim / a.1, b.2, c.3 /;

scalar param_0dim / 5 /;
parameter param_1dim(set_1dim) / a 1, b 2, c 3 /;
parameter param_2dim(set_1dim, set_1dim) / a.b 1, b.c 2, c.a 3 /;
parameter param_3dim(set_1dim, set_1dim, set_1dim) / a.a.a 1, b.b.b 2, c.c.c 3/;
parameter param_4dim(set_1dim, set_1dim, set_1dim, set_1dim) / a.b.a.b 1, b.c.b.c 2, c.a.c.a 3/;

execute_unload 'test1.gdx';
execute_unload 'test2.gdx';
