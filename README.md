# Áp dụng Reinforcement Learning cho Game Pacman
Project sử dụng hai phương pháp Q-Learning và SARSA để tạo ra AI Pacman có thể tránh né các con Ghost và ăn với số đốt tối ưu nhất. Cuối cùng đưa ra so sánh giũa hai phương pháp học tăng cường trên.
## Getting Started
Project được viết bằng ngôn ngữ Python 3.6 và sử dụng những thư viện cơ sở có sẵn. Chon nên rất dễ cài đặt và sử dụng.
### Prerequisites
```
Python >3.5
```
### Installing
các thư viện bổ sung:
+ opencv
+ numpy
+ matplotlib.pyplot
+ termcolor
Đẻ cái các thư viện bạn chuyển terminal đến thư mục chính và:
Trên window chạy các câu lệnh cài đặt:
```
pip3 install numpy
pip3 install opencv
pip3 install matplotlib
pip3 install termcolor
```
Trên ubuntu chạy các câu lệnh cài đặt:
```
sudo pip3 install numpy
sudo pip3 install opencv
sudo pip3 install matplotlib
sudo pip3 install termcolor
```
Hoặc trên môi trường anaconda:
```
pip install numpy
...
```
## Running the tests
- Khởi tạo một quá trình train mới:

Bạn edit file config thành:
```
INIT_Q_Learning = True
INIT_Q_Sarsa = True
```
Sau đó bạn có thể chạy file trainning.py (Q_Learning) và trainning_sarsa.py (SARSA) để quá trình khởi tạo bắt đầu:
```
python3 trainning.py
```
```
python3 trainning_sarsa.py
```
(câu lệnh trong terminal)

Sau khi run file trên giá trị khởi tạo sẽ có dạng như sau:
```
Episode 0: rate: 1.0 - dot: 0 - rewards: -100000
Saved Qtable
Max dot:  3  Min dot:  3
Ave D=  3.0  S= 3.0  
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 ...
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
 ```
 (trong trainning.py)
 - Bắt đầu quá trình train:
 
 Bạn edit file config thành:
```
INIT_Q_Learning = False
INIT_Q_Sarsa = False
```
Sau đó bạn chạy lại file trainning.py (Q_Learning) và trainning_sarsa.py (SARSA) để quá trình train bắt đầu:

Nếu quá trình train đã được bắt đầu thì terminal sẽ có thông tin dạng như sau:
```
Episode 1: rate: 1.0 - dot: 1 - rewards: -99988
Episode 2: rate: 0.9995051237293776 - dot: 3 - rewards: -99966
Episode 3: rate: 0.9990104948350412 - dot: 4 - rewards: -99952
Episode 4: rate: 0.9985161131933338 - dot: 4 - rewards: -99954
Episode 5: rate: 0.9980219786806598 - dot: 7 - rewards: -99916
...
```
- Xem thông tin train:
```
run file: view-graph.py
```
- Test game:
```
run file: check_q.py
```
(* Lưu ý trong file check_q.py : để test game với Q-Learning bạn thay:
```
check(1, 'S') -> check(1, 'Q')
```
, hoặc ngược lại)

Một trận đấu sẽ được hiển thị trên terminal như sau:

![alt text](https://i.imgur.com/Qm94651.png)

## Publish the result
Dưới đây là kết quả khi train giữa Q-Learning vs SARSA:
![alt text](https://i.imgur.com/NGl5tYh.png)
## Environment
Môi trường train hiện tại:
```
11 20
####################
#....#........#....#
#.##.#.######.#.##.#
#.#..............#.#
#.#.##.##++##.##.#.#
#......#++++#......#
#.#.##.######.##.#.#
#.#..............#.#
#.##.#.######.#.##.#
#....#...+....#....#
####################
0
2
6 10
6 11
10 10
```
+ Môi trường là một map trung bình có hai con Ghost được thiết kế di chuyển ngẫu nhiên trong map.
+ Môi trường cung cấp trạng thái của con Pacman dưới các trích chọn đặc trưng sau:
```
- Bốn ô xung quang là tường hay không?
- Trong 8 ô gần nhất Ghost có xuất hiện hay không? và hướng gần nhât Ghost xuất hiện.
- Pacman có đang rơi vào trạng thái bẫy hay không?
- Hướng đi ngắn nhất tới dot.
```
