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
