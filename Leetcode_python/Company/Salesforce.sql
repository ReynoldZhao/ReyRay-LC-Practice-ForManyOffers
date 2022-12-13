create table test;
use test;

create table mytable (
    id int not null AUTO_INCREMENT,
    col1 int not null default 1,
    col2 varchar(45) null,
    col3 date null,
    primary key (`id`)
)

alter table mytable
add col char(20)

alter table mytable
drop column col

drop table mytable

insert into mytable(col1, col2)
values(val1, val2);

insert into mytable1(col1, col2)
select col1, col2
from mytable2

create table newtable as
select * from mytable
