SELECT name, population, area
FROM World
where population >= 25000000 or area >= 3000000

SELECT
IFNULL((SELECT DISTINCT salary
FROM Employee
ORDER BY salary DESC
LIMIT 1 OFFSET 1), NULL) as SecondHighestSalary

SELECT A.name as 'Employee'
FROM Employee as A, Employee as B
WHERE
    A.manageId = B.id
    AND
    A.salary > B.salary

SELECT A.name as 'Employee'
FROM Employee as A JOIN Employee as B
ON A.managerId = B.id
where A.salary > B.salary

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE M INT;
set M = N - 1
  RETURN (
      # Write your MySQL query statement below.
      select ifnull((select DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT M, 1), NULL)
  );
END

select FirstName, LastName, City, State
from Person LEFT JOIN Address
on Person.personId = Address.personId

select DISTINCT l1.num as ConsecutiveNums
FROM
    log l1,
    log l2,
    log l3,
WHERE
    l1.id = l2.id - 1
    AND l2.id = l3.id - 1
    And l1.num = l2.num
    And l2.num = l3.num

SELECT Name AS 'Customers'
FROM Customers c
LEFT JOIN Orders o
ON c.Id = o.CustomerId
WHERE o.CustomerId IS NULL

SELECT D.Name AS Department ,E.Name AS Employee ,E.Salary
FROM
    Employee E,
    (SELECT departmentId, max(salary) as maxS FROM Employee GROUP BY departmentId) T,
    Department D
WHERE E.DepartmentId = T.DepartmentId 
  AND E.Salary = T.max
  AND E.DepartmentId = D.id

delete p1
FROM 
Person p1,
Person p2
WHERE p1.id not IN (select min(id) as id from Person GROUP BY email)

SELECT ROUND(AVG(CASE WHEN low_fats = 'Y' OR recyclable = 'Y' THEN 1 ELSE 0 END),2) AS PERCENTAGE

select C.party, count(C.id)
from
    (select candidate_id as won_c_id, max(votes) as max_v from results group by constituency_id) T1
    Candidates C
where C.id = T1.won_c_id
group by C.party

select P.category, P.product_id, P.discount
FROM Product P
    (select product_id as pid, min(discount) as min_dis, category as cat
    from Product
    GROUP BY category
    ORDER BY product_id des
    LIMIT 1) T
where P.product_id = T.pid
order by category ASC



select country.coutry_name, city.city_name, T.customer_number

FROM
    country
    city
    (select count(customer_id) as customer_number, city_id as cityId
    from customer
    GROUP BY city_id) T

where 
    T.cityId = city.id and
    city.country_id = country.id and
    T.customer_number > (SELECT (SELECT COUNT(id) from customer) / (SELECT COUNT(id) from city))

select s1.score,
(select count(DISTINCT score)
from Scores S2
where s1.score >= s2.score) as 'Rank'
from Score s1
ORDER BY s1.score DESC

delete p2
from Person p1 JOIN Person p2
on p1.Email = p2.Email
where p1.id < p2.id

create function getNthHighestSalary(N INT) RETURNS INT
begin
declare M int;
set M = N - 1;
    return (
        SELECT IFNULL((SELECT DISTINCT salary from Employee ORDER BY Salary Desc LIMIT M, 1), NULL)
    );
END

select player_id, event_date as first_login
from Activity
GROUP BY player_id
ORDER BY event_date
LIMIT 1