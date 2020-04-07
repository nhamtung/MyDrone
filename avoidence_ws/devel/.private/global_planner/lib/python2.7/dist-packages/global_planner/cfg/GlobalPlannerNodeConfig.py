## *********************************************************
##
## File autogenerated for the global_planner package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'upper': 'DEFAULT', 'lower': 'groups', 'srcline': 246, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'cstate': 'true', 'parentname': 'Default', 'class': 'DEFAULT', 'field': 'default', 'state': True, 'parentclass': '', 'groups': [], 'parameters': [{'srcline': 291, 'description': 'Minimum planned altitude', 'max': 10, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'min_altitude_', 'edit_method': '', 'default': 1, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 291, 'description': 'Maximum planned altitude', 'max': 50, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'max_altitude_', 'edit_method': '', 'default': 10, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 291, 'description': 'Maximum risk allowed per cells', 'max': 1.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'max_cell_risk_', 'edit_method': '', 'default': 0.2, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Cost of turning', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'smooth_factor_', 'edit_method': '', 'default': 20.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Cost of changing between horizontal and vertical direction', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'vert_to_hor_cost_', 'edit_method': '', 'default': 3.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Cost of crashing', 'max': 1000.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'risk_factor_', 'edit_method': '', 'default': 500.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'The effect of the risk of neighboring cells', 'max': 1.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'neighbor_risk_flow_', 'edit_method': '', 'default': 1.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'The cost of unexplored space', 'max': 0.01, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'expore_penalty_', 'edit_method': '', 'default': 0.005, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Cost of ascending 1m', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'up_cost_', 'edit_method': '', 'default': 5.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Cost of descending 1m', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'down_cost_', 'edit_method': '', 'default': 1.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Time it takes to return a new path', 'max': 1.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'search_time_', 'edit_method': '', 'default': 0.5, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'The minimum overestimation for heuristics', 'max': 1.5, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'min_overestimate_factor_', 'edit_method': '', 'default': 1.03, 'level': 0, 'min': 1.0, 'type': 'double'}, {'srcline': 291, 'description': 'The minimum overestimation for heuristics', 'max': 5.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'max_overestimate_factor_', 'edit_method': '', 'default': 2.0, 'level': 0, 'min': 1.0, 'type': 'double'}, {'srcline': 291, 'description': 'Maximum number of iterations', 'max': 10000, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'max_iterations_', 'edit_method': '', 'default': 2000, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 291, 'description': "Don't bother trying to find a path if the exact goal is occupied", 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'goal_must_be_free_', 'edit_method': '', 'default': True, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 291, 'description': 'The current yaw affects the pathfinding', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'use_current_yaw_', 'edit_method': '', 'default': True, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 291, 'description': 'Use non underestimating heuristics for risk', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'use_risk_heuristics_', 'edit_method': '', 'default': True, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 291, 'description': 'Use non underestimating heuristics for speedup', 'max': True, 'cconsttype': 'const bool', 'ctype': 'bool', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'use_speedup_heuristics_', 'edit_method': '', 'default': True, 'level': 0, 'min': False, 'type': 'bool'}, {'srcline': 291, 'description': 'The altitude of clicked goals', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'clicked_goal_alt_', 'edit_method': '', 'default': 3.5, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Minimum allowed distance from path end to goal', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'clicked_goal_radius_', 'edit_method': '', 'default': 1.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Maximum number of iterations to simplify a path', 'max': 100, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'simplify_iterations_', 'edit_method': '', 'default': 1, 'level': 0, 'min': 0, 'type': 'int'}, {'srcline': 291, 'description': 'The allowed cost increase for simplifying an edge', 'max': 2.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'simplify_margin_', 'edit_method': '', 'default': 1.01, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Size of a cell, should be divisable by the OctoMap resolution', 'max': 2.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'CELL_SCALE', 'edit_method': '', 'default': 1.0, 'level': 2, 'min': 0.5, 'type': 'double'}, {'srcline': 291, 'description': 'Maximum length of edge between two Cells', 'max': 10.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'SPEEDNODE_RADIUS', 'edit_method': '', 'default': 5.0, 'level': 4, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Change search mode', 'max': '', 'cconsttype': 'const char * const', 'ctype': 'std::string', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'default_node_type_', 'edit_method': "{'enum_description': 'Change search mode', 'enum': [{'srcline': 40, 'description': 'Normal node', 'srcfile': '/home/nhamtung/TungNV/MyDrone/avoidence_ws/src/avoidance/global_planner/cfg/GlobalPlannerNode.cfg', 'cconsttype': 'const char * const', 'value': 'Node', 'ctype': 'std::string', 'type': 'str', 'name': 'Node'}, {'srcline': 41, 'description': 'No smooth cost', 'srcfile': '/home/nhamtung/TungNV/MyDrone/avoidence_ws/src/avoidance/global_planner/cfg/GlobalPlannerNode.cfg', 'cconsttype': 'const char * const', 'value': 'NodeWithoutSmooth', 'ctype': 'std::string', 'type': 'str', 'name': 'NodeWithoutSmooth'}, {'srcline': 42, 'description': 'Search with speed', 'srcfile': '/home/nhamtung/TungNV/MyDrone/avoidence_ws/src/avoidance/global_planner/cfg/GlobalPlannerNode.cfg', 'cconsttype': 'const char * const', 'value': 'SpeedNode', 'ctype': 'std::string', 'type': 'str', 'name': 'SpeedNode'}]}", 'default': 'SpeedNode', 'level': 4, 'min': '', 'type': 'str'}], 'type': '', 'id': 0}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

GlobalPlannerNode_Node = 'Node'
GlobalPlannerNode_NodeWithoutSmooth = 'NodeWithoutSmooth'
GlobalPlannerNode_SpeedNode = 'SpeedNode'
