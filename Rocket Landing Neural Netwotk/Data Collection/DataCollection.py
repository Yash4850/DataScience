
class DataCollection:

    def __init__(self):
        self.data_file = open("ce889_dataCollection.csv", "a")
        self.data_file.close()
        self.buffer = []

    def get_input_row(self, lander, surface, controller):
        # inputs
        current_velocity = lander.velocity
        current_speed = current_velocity.length()
        current_angle = lander.current_angle
        x_target = surface.centre_landing_pad[0] - lander.position.x
        y_target = surface.centre_landing_pad[1] - lander.position.y
        dist_to_surface = surface.polygon_rect.topleft[1] - lander.position.y

        # create comma separated string row
        input_row = str(current_speed)+"," + \
                    str(current_velocity.x) + "," + \
                    str(current_velocity.y) + "," + \
                    str(current_angle) + "," + \
                    str(x_target) + "," + \
                    str(y_target) + "," + \
                    str(dist_to_surface)

        return input_row

    def save_current_status(self, input_row, lander, surface, controller):
        # open file
        # input_row = self.get_input_row(lander, surface, controller)

        # outputs
        thrust = 0
        if (controller.is_up()):
            thrust = 1
        new_vel_y = lander.velocity.y
        
        turning = [0, 0]
        if (controller.is_left()):
            turning = [1,0]
        elif (controller.is_right()):
            turning = [0,1]
        new_angle = lander.current_angle
        # add output values to the string input row
        status_row = input_row + "," + \
                    str(thrust) + "," + \
                    str(new_vel_y) + "," + \
                    str(new_angle) + "," + \
                    str(turning[0]) + "," + str(turning[1]) + "\n"
        
        # save comma separated row in the file
        self.buffer.append(status_row)


    def write_to_file(self):
        self.data_file = open("ce889_dataCollection.csv", "a")
        for row in self.buffer:
            self.data_file.write(row)
        self.data_file.close()

    def reset(self):
        self.buffer = []