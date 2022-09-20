#include <AFMotor.h>
char command;

AF_DCMotor motor1(1);  
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);  
AF_DCMotor motor4(4);

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    command = Serial.read();
    stop_();
    switch(command){
      case 'F':  
        forward();
        break;
      case 'B':  
        back();
        break;
      case 'L':  
        left();
        break;
      case 'R':
        right();
        break;
      case ' ':
        stop_();
        break;
    }
  }
}

void forward()
{
  motor1.setSpeed(255); // Define maximum velocity
  motor1.run(FORWARD);  // Rotate the motor clockwise
  motor2.setSpeed(255); // Define maximum velocity
  motor2.run(FORWARD);  // Rotate the motor clockwise
  motor3.setSpeed(255); // Define maximum velocity
  motor3.run(FORWARD);  // Rotate the motor clockwise
  motor4.setSpeed(255); // Define maximum velocity
  motor4.run(FORWARD);  // Rotate the motor clockwise
}

void back()
{
  motor1.setSpeed(255); // Define maximum velocity
  motor1.run(BACKWARD); // Rotate the motor anti-clockwise
  motor2.setSpeed(255); // Define maximum velocity
  motor2.run(BACKWARD); // Rotate the motor anti-clockwise
  motor3.setSpeed(255); // Define maximum velocity
  motor3.run(BACKWARD); // Rotate the motor anti-clockwise
  motor4.setSpeed(255); // Define maximum velocity
  motor4.run(BACKWARD); // Rotate the motor anti-clockwise
}

void left()
{
  motor1.setSpeed(255); // Define maximum velocity
  motor1.run(BACKWARD); // Rotate the motor anti-clockwise
  motor2.setSpeed(255); // Define maximum velocity
  motor2.run(BACKWARD); // Rotate the motor anti-clockwise
  motor3.setSpeed(255); // Define maximum velocity
  motor3.run(FORWARD);  // Rotate the motor clockwise
  motor4.setSpeed(255); // Define maximum velocity
  motor4.run(FORWARD);  // Rotate the motor clockwise
}

void right()
{
  motor1.setSpeed(255); // Define maximum velocity
  motor1.run(FORWARD);  // Rotate the motor clockwise
  motor2.setSpeed(255); // Define maximum velocity
  motor2.run(FORWARD);  // Rotate the motor clockwise
  motor3.setSpeed(255); // Define maximum velocity
  motor3.run(BACKWARD); // Rotate the motor anti-clockwise
  motor4.setSpeed(255); // Define maximum velocity
  motor4.run(BACKWARD); // Rotate the motor anti-clockwise
} 

void stop_()
{
  motor1.setSpeed(0);   // Define minimum velocity
  motor1.run(RELEASE);  // Stop the motor when release the button
  motor2.setSpeed(0);   // Define minimum velocity
  motor2.run(RELEASE);  // Rotate the motor clockwise
  motor3.setSpeed(0);   // Define minimum velocity
  motor3.run(RELEASE);  // Stop the motor when release the button
  motor4.setSpeed(0);   // Define minimum velocity
  motor4.run(RELEASE);  // Stop the motor when release the button
}
