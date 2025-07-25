#include "Controller.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <thread>
#include "utilities.h"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

Controller::Controller(const std::string &net_interface)
{
        YAML::Node yaml_node = YAML::LoadFile("../../configs/g1.yaml");

	leg_joint2motor_idx = yaml_node["leg_joint2motor_idx"].as<std::vector<float>>();
        kps = yaml_node["kps"].as<std::vector<float>>();
        kds = yaml_node["kds"].as<std::vector<float>>();
	default_angles = yaml_node["default_angles"].as<std::vector<float>>();
	arm_waist_joint2motor_idx = yaml_node["arm_waist_joint2motor_idx"].as<std::vector<float>>();
	arm_waist_kps = yaml_node["arm_waist_kps"].as<std::vector<float>>();
	arm_waist_kds = yaml_node["arm_waist_kds"].as<std::vector<float>>();
	arm_waist_target = yaml_node["arm_waist_target"].as<std::vector<float>>();
	ang_vel_scale = yaml_node["ang_vel_scale"].as<float>();
	dof_pos_scale = yaml_node["dof_pos_scale"].as<float>();
	dof_vel_scale = yaml_node["dof_vel_scale"].as<float>();
	action_scale = yaml_node["action_scale"].as<float>();
	cmd_scale = yaml_node["cmd_scale"].as<std::vector<float>>();
	num_actions = yaml_node["num_actions"].as<float>();
	num_obs = yaml_node["num_obs"].as<float>();
	max_cmd = yaml_node["max_cmd"].as<std::vector<float>>();

	obs.setZero(num_obs);
	act.setZero(num_actions);

	module = torch::jit::load("../../../pre_train/g1/motion.pt");

	unitree::robot::ChannelFactory::Instance()->Init(0, net_interface);

	lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
	lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(TOPIC_LOWSTATE));

	lowcmd_publisher->InitChannel();
	lowstate_subscriber->InitChannel(std::bind(&Controller::low_state_message_handler, this, std::placeholders::_1));

	while (!mLowStateBuf.GetDataPtr())
	{
		usleep(100000);
	}
	
	low_cmd_write_thread_ptr = unitree::common::CreateRecurrentThreadEx("low_cmd_write", UT_CPU_ID_NONE, 2000, &Controller::low_cmd_write_handler, this);
	std::cout << "Controller init done!\n";
}

void Controller::zero_torque_state()
{
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	std::cout << "zero_torque_state, press start\n";
	while (!joy.btn.components.start)
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();

		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.q() = 0;
			cmd.dq() = 0;
			cmd.kp() = 0;
			cmd.kd() = 0;
			cmd.tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::move_to_default_pos()
{
	std::cout << "move_to_default_pos, press A\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	auto low_state = mLowStateBuf.GetDataPtr();	
	std::array<float, 35> jpos;
	for (int i = 0; i < 35; i++)
	{
		jpos[i] = low_state->motor_state()[i].q();
	}

	int num_steps = 100;
	int count = 0;

	while (count <= num_steps || !joy.btn.components.A) 
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		float phase = std::clamp<float>(float(count++) / num_steps, 0, 1);
		
		// leg
		for (int i = 0; i < 12; i++)
		{
			low_cmd->motor_cmd()[i].q() = (1 - phase) * jpos[i] + phase * default_angles[i];
			low_cmd->motor_cmd()[i].kp() = kps[i];
			low_cmd->motor_cmd()[i].kd() = kds[i];
			low_cmd->motor_cmd()[i].tau() = 0.0;
			low_cmd->motor_cmd()[i].dq() = 0.0;
		}

		// waist arm
		for (int i = 12; i < 29; i++)
		{
			low_cmd->motor_cmd()[i].q() = (1 - phase) * jpos[i] + phase * arm_waist_target[i - 12];
			low_cmd->motor_cmd()[i].kp() = arm_waist_kps[i - 12];
			low_cmd->motor_cmd()[i].kd() = arm_waist_kds[i - 12];
			low_cmd->motor_cmd()[i].tau() = 0.0;
			low_cmd->motor_cmd()[i].dq() = 0.0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::run()
{
	std::cout << "run controller, press select\n";

	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	float period = .8;
	float time = 0;

	while (!joy.btn.components.select)
	{
		auto low_state = mLowStateBuf.GetDataPtr();
		// obs
		Eigen::Matrix3f R = Eigen::Quaternionf(low_state->imu_state().quaternion()[0], low_state->imu_state().quaternion()[1], low_state->imu_state().quaternion()[2], low_state->imu_state().quaternion()[3]).toRotationMatrix();

		for (int i = 0; i < 3; i++)
		{
			obs(i) = ang_vel_scale * low_state->imu_state().gyroscope()[i];
			obs(i + 3) = -R(2, i);
		}

		if (obs(5) > 0)
		{
			break;
		}

		obs(6) = joy.ly * max_cmd[0] * cmd_scale[0];
		obs(7) = joy.lx * -1 * max_cmd[1] * cmd_scale[1];
		obs(8) = joy.rx * -1 * max_cmd[2] * cmd_scale[2];

		for (int i = 0; i < 12; i++)
		{
			obs(9 + i) = (low_state->motor_state()[i].q() - default_angles[i]) * dof_pos_scale;
			obs(21 + i) = low_state->motor_state()[i].dq() * dof_vel_scale;
		}
		obs.segment(33, 12) = act;

		float phase = std::fmod(time / period, 1);
		time += .02;
		obs(45) = std::sin(2 * M_PI * phase);
		obs(46) = std::cos(2 * M_PI * phase);

		// policy forward
		torch::Tensor torch_tensor = torch::from_blob(obs.data(), {1, obs.size()}, torch::kFloat).clone();
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch_tensor);
		torch::Tensor output_tensor = module.forward(inputs).toTensor();
		std::memcpy(act.data(), output_tensor.data_ptr<float>(), output_tensor.size(1) * sizeof(float));

		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		// leg
		for (int i = 0; i < 12; i++)
		{
			low_cmd->motor_cmd()[i].q() = act(i) * action_scale + default_angles[i];
			low_cmd->motor_cmd()[i].kp() = kps[i];
			low_cmd->motor_cmd()[i].kd() = kds[i];
			low_cmd->motor_cmd()[i].dq() = 0;
			low_cmd->motor_cmd()[i].tau() = 0;
		}

		// waist arm
		for (int i = 12; i < 29; i++)
		{
			low_cmd->motor_cmd()[i].q() = arm_waist_target[i - 12];
			low_cmd->motor_cmd()[i].kp() = arm_waist_kps[i - 12];
			low_cmd->motor_cmd()[i].kd() = arm_waist_kds[i - 12];
			low_cmd->motor_cmd()[i].dq() = 0;
			low_cmd->motor_cmd()[i].tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::damp()
{
	std::cout << "damping\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	while (true)
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.kp() = 0;
			cmd.kd() = 8;
			cmd.dq() = 0;
			cmd.tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}


void Controller::low_state_message_handler(const void *message)
{
	unitree_hg::msg::dds_::LowState_* ptr = (unitree_hg::msg::dds_::LowState_*)message;
	mLowStateBuf.SetData(*ptr);
	std::memcpy(&joy, ptr->wireless_remote().data(), ptr->wireless_remote().size() * sizeof(uint8_t));
}

void Controller::low_cmd_write_handler()
{
	if (auto lowCmdPtr = mLowCmdBuf.GetDataPtr())
	{
		lowCmdPtr->mode_machine() = mLowStateBuf.GetDataPtr()->mode_machine();
		lowCmdPtr->mode_pr() = 0;
		for (auto &cmd : lowCmdPtr->motor_cmd())
		{
			cmd.mode() = 1;
		}
		lowCmdPtr->crc() = crc32_core((uint32_t*)(lowCmdPtr.get()), (sizeof(unitree_hg::msg::dds_::LowCmd_) >> 2) - 1);
		lowcmd_publisher->Write(*lowCmdPtr);
	}
}
