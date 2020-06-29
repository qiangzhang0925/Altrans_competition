"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
grade:581.8-->单个ack测量带宽
grade:582.43-->3个间隔ack测量带宽
"""
import math

from simple_emulator import PccEmulator, CongestionControl
from simple_emulator.objects import sender
from simple_emulator import analyze_pcc_emulator, plot_rate

from simple_emulator import Packet_selection

# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno

from simple_emulator import cal_qoe

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'


class MySolution(Packet_selection, Reno):

    def __init__(self):
        # base parameters in CongestionControl
        self._input_list = []
        self.cwnd = 1
        self.send_rate = float("inf")
        self.pacing_rate = float("inf")
        self.call_nums = 0
        self.rev_nums = 0
        self.inflight = 0
        self.count = 0 #带宽探测计数
        self.num = 0
        self.sp = 0
        self.last_delta = 0

        self.min_latency = None
        self.BtlBW = None
        self.gain = [5/4, 3/4, 1, 1, 1, 1, 1, 1]

        # for bbr
        self.curr_state = "slow_start"
        self.states = ["slow_start", "drain", "prob_bw"]
        self.ack_nums = 0

        self.cur_time = -1

    def select_packet(self, cur_time, packet_queue):
        """
        The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
        The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#packet_selectionpy.
        :param cur_time: float
        :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
        :return: int
        """
        def is_better(packet):
            best_block_create_time = best_packet.block_info["Create_time"]
            packet_block_create_time = packet.block_info["Create_time"]
            best_time_left = best_packet.block_info["Deadline"] - (cur_time - best_block_create_time)
            packet_time_left = packet.block_info["Deadline"] - (cur_time - packet_block_create_time)
            best_size_left = best_packet.block_info["Size"] / 1480 - best_packet.offset
            packet_size_left = packet.block_info["Size"] / 1480 - packet.offset
            # if packet is miss ddl
            if (cur_time - packet_block_create_time) >= packet.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
                return True
            # 同一个block按序发送
            if best_packet.block_info["Block_id"] == packet.block_info["Block_id"]:
                return best_packet.offset > packet.offset
            else:
                if best_packet.block_info["Priority"] == packet.block_info["Priority"]:
                    return best_time_left / best_size_left < packet_time_left / packet_size_left
                else:
                    return best_packet.block_info["Priority"] > packet.block_info["Priority"]

        best_packet_idx = -1
        best_packet = None
        block_list = []

        for idx, item in enumerate(packet_queue):
            if item.block_info["Block_id"] not in block_list:
                block_list.append(item.block_info["Block_id"])
                if best_packet is None or is_better(item):
                    best_packet_idx = idx
                    best_packet = item

        return best_packet_idx

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def cc_trigger(self, data):
        self.rev_nums = len(self._input_list)
        event_type = data["event_type"]
        event_time = data["event_time"]
        if event_type == 'D': return
        rtt = data["packet_information_dict"]["Latency"] + data["packet_information_dict"]["Pacing_delay"]
        sendtime = event_time - rtt
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.inflight = data["packet_information_dict"]["Extra"]["inflight"]
        flag = 1
        while self.rev_nums and flag == 1:
            data1 = self._input_list[self.rev_nums - 1]
            if data1["event_type"] == 'F' and data1["event_time"] >= sendtime:
                self.ack_nums += 1
            if data1["event_time"] < sendtime: flag = 0
            self.rev_nums += -1
        bandwith = self.ack_nums / rtt
        self.ack_nums = 0

        if (self.BtlBW is None) or (bandwith > self.BtlBW):
            self.BtlBW = bandwith

        #即时的带宽时延积
        BDP = self.BtlBW * self.min_latency

        if event_type == EVENT_TYPE_FINISHED:
            # Ref 1
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            #startup 阶段
            if self.curr_state == self.states[0]:
                if BDP > self.inflight:
                    self.cwnd *= 2
                    if self.pacing_rate == float("inf"):
                        self.pacing_rate = 2*self.BtlBW
                    self.pacing_rate *= 2
                else:
                    self.sp +=1
                    self.cwnd *=2
                    self.pacing_rate *=2
                    if self.sp == 2: self.curr_state = self.states[1]

            #排空阶段
            elif self.curr_state == self.states[1]:
                if self.inflight <= BDP:
                    self.curr_state = self.states[2]
                else:
                    self.cwnd *= 0.5
                    self.pacing_rate *= 0.5
                    if self.cwnd < 1:
                        self.cwnd =1
            #带宽探测：
            elif self.curr_state == self.states[2]:
                self.num += 1
                if self.num == 5:
                    self.count += 1
                    self.pacing_rate = self.gain[self.count - 1] * self.BtlBW
                    self.cwnd = 2*BDP
                    self.num = 0
                if self.count == 8:
                    self.count = 0
                    # if self.inflight > 0.5*BDP:
                    #     self.BtlBW = None
                    #     self.curr_state = self.states[1]

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd" : self.cwnd,
                # "send_rate" : self.send_rate,
                "pacing_rate": self.pacing_rate
            }
        return None


if __name__ == '__main__':

    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"

    # Use the object you created above
    my_solution = MySolution()

    # Create the emulator using your solution
    # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#constant
    emulator = PccEmulator(
        block_file=["traces/data_video.csv", "traces/data_audio.csv"],
        trace_file="traces/trace.txt",
        solution=my_solution,
        USE_CWND=True,
        ENABLE_LOG=True
    )

    # Run the emulator and you can specify the time for the emualtor's running.
    # It will run until there is no packet can sent by default.
    emulator.run_for_dur(15)

    # print the debug information of links and senders
    emulator.print_debug()
    # Output the picture of pcc_emulator-analysis.png
    # You can get more information from https://github.com/Azson/DTP-emulator/tree/pcc-emulator#pcc_emulator-analysispng.
    analyze_pcc_emulator(log_packet_file, file_range="all")

    # Output the picture of rate_changing.png
    # You can get more information from https://github.com/Azson/DTP-emulator/tree/pcc-emulator#cwnd_changingpng
    plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all", sender=[1])

    result=cal_qoe()

    print(result)
