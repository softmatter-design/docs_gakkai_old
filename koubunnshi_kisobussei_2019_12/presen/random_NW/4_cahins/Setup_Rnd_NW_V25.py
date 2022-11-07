#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################
## 計算条件
##################
# 使用するCognacのバージョンを入れてください。
ver_cognac = "cognac10"
blank_udf = 'cognac100.udf'
# 計算に使用するコア数
core = 7
############
# リスタートの有無
restart = 0
################################################################
# 基本ネットワーク関連の設定
################################################################
# 設定したい分岐数
n_chains = 4
#################
# ストランドの長さ
n_strand = 50
##########################
# 一辺当たりの単位ユニット数
n_cell = 5
#################
# ストランドの側鎖
n_sc = 0
###################################
# ファントムネットワークの場合の多重度
multi_phantom = 4
################################################################
# トポロジー変換に関する設定
################################################################
# プレ探索の条件
pre_sampling = 1000
pre_try = 1000
# 繰り返しサンプリング数
n_sampling = 1000
# 最小構造探索の繰り返し数
n_try = 1000
####################
# ランダム化のフラグ
f_rand = 0
# 計算時の多重度
f_pool = 1
################################################################
# ヒストグラム関連の設定
################################################################
# ヒストグラムの分割数
hist_bins = 500
################################################################
# ネットワークのタイプ
################################################################
nw_type = "NoLJ_Harmonic"
# nw_type = "LJ_Harmonic_simple"
# nw_type = "LJ_Harmonic"
# nw_type = "KG"
# nw_type = "Atr_LJ_w_Ang"
################################################################
def main():
	#####################
	# 全体の初期状態を設定
	#####################
	if restart == 1:
		global n_chains, n_cell
		#
		print("#################### \n Restart Calculation")
		#
		read_n_chains, read_n_cell, read_file_path = read_cond()
		if read_n_chains != n_chains:
			print("Number of Chains is different from read data!")
		n_chains = read_n_chains
		n_cell = read_n_cell
		#
	init = InitialSetup(n_strand, n_sc, n_chains, n_cell, nw_type, multi_phantom)
	# 基本となるデータを取得
	target_name, multi_nw, density, system_size, a_cell = init.get_base_data()
	# 基準となる8_Chainネットワークを設定
	init_8ch_dic, jp_xyz_dic, atom_jp, n_jp, vector_dic = init.make_8chain_dic()

	##################################################################
	# トポロジーの異なるネットワークを探索して、代数的連結性の分布関数を策定
	##################################################################
	mod = ModifyTop(init_8ch_dic, n_chains, n_jp, n_cell, pre_try, pre_sampling, n_try, n_sampling, f_rand, f_pool, multi_nw, hist_bins)
	# 任意の多重度のネットワークポリマーの初期状態を探索
	if restart != 1:
		top_dic_list = mod.top_search()
	else:
		top_dic_list = mod.top_select(read_file_path)

	###########################################
	# ターゲットとなるネットワーク全体の辞書を設定。
	###########################################
	setup = SetUp(top_dic_list, vector_dic, atom_jp, jp_xyz_dic, n_strand, n_sc)
	# ターゲットとなるネットワーク全体の辞書を設定。
	calcd_data_dic_list = setup.make_data_dic()

	##################
	# Initi_UDF の作成
	##################
	makebase = MakeBaseUDF(calcd_data_dic_list, blank_udf, multi_nw, system_size, a_cell)
	#-----base_udfを作成
	base_udf = makebase.makeudf()

	####################
	# バッチファイルを作成
	####################
	make_batch = MakeBatch(target_name, base_udf, ver_cognac, core, nw_type)
	# ファイル名を設定し、バッチファイルを作成
	make_batch.setup_all()

################################################################################
import numpy as np
import copy
import random
import platform
import subprocess
import sys
import statistics
from UDFManager import UDFManager
import os
import pickle
from multiprocessing import Pool
import multiprocessing
################################################################################
# 
################################################################################
#----- File Select
def read_cond():
	param = sys.argv
	if len(param) == 1:
		print("in Restart Calculation: usage: python", param[0], "Reading Dir")
		exit(1)
	if not os.path.exists(os.path.join(param[1], 'init.pickle')):
		print(param[1], "not exists.")
		exit(1)
	else:
		read_file_path = param[1]
		n_chains = int(param[1].split('_')[0])
		n_cell = int(param[1].split('_')[2])
	return n_chains, n_cell, read_file_path
#
class InitialSetup:
	def __init__(self, n_strand, n_sc, n_chains, n_cell, nw_type, multi_phantom):
		self.n_strand = n_strand
		self.n_sc = n_sc
		self.n_chains = n_chains
		self.nw_type = nw_type
		self.n_cell = n_cell
		self.multi_phantom = multi_phantom
		# ユニットセルでの、jp およびサブチェインの始点と終点のXYZを設定
		self.jp_xyz = [
				[0.,0.,0.], 
				[0.5, 0.5, 0.5]
				]
		self.strand_se_xyz = [
					[[0.5, 0.5, 0.5], [0, 0, 0]],
					[[0.5, 0.5, 0.5], [1, 0, 0]],
					[[0.5, 0.5, 0.5], [0, 1, 0]],
					[[0.5, 0.5, 0.5], [1, 1, 0]],
					[[0.5, 0.5, 0.5], [0, 0, 1]],
					[[0.5, 0.5, 0.5], [1, 0, 1]],
					[[0.5, 0.5, 0.5], [0, 1, 1]],
					[[0.5, 0.5, 0.5], [1, 1, 1]]
					]
		self.start_jp = [0.5, 0.5, 0.5]
	#######################
	# 基本となるデータを取得
	def get_base_data(self):
		multi_nw, density, system_size, a_cell = self.calc_conditions()
		target_name = self.set_target_name(multi_nw)
		return target_name, multi_nw, density, system_size, a_cell

	###################
	# target_nameを決める。
	def set_target_name(self, multi_nw):
		target_name = str(self.nw_type ) + "_" + str(self.n_chains) + '_Chains_N_' + str(self.n_strand) + "_C_" + str(self.n_cell) + "_M_" + str(multi_nw)
		return target_name

	###########################
	# 各種の条件を算出する
	def calc_conditions(self):
		if self.nw_type == "NoLJ_Harmonic":
			bond = 0.97
			c_inf = 1.0
		elif self.nw_type == "KG" or self.nw_type == "LJ_Harmonic"  or self.nw_type == "LJ_Harmonic_simple":
			bond = 0.97
			c_inf = 1.7
			target_density = 0.85
		elif self.nw_type == "Atr_LJ_w_Ang":
			bond = 0.97
			c_inf = 1.0
			target_density = 0.85
		#
		print("#########################################")
		print("対象となるネットワークタイプ\t", self.nw_type)
		print("分岐数\t\t\t\t", self.n_chains)
		print("ストランド中のセグメント数\t", self.n_strand)
		print("一辺当たりの単位ユニット数\t", self.n_cell)
		print("#########################################")
		#
		e2e = bond*((self.n_strand + 1)*c_inf)**0.5						# 理想鎖状態での末端間距離
		n_beads_unit = 2 + self.n_strand*(1 + self.n_sc)*self.n_chains	# ユニットセル当たりの粒子数
		a_cell = (2*3**0.5)*e2e/3										# 理想鎖状態でのユニットセル長
		init_dens = n_beads_unit/a_cell**3								# 理想鎖状態でのユニットセル長
		nu = self.n_chains/a_cell**3									# ストランドの数密度
		if self.nw_type == "KG" or self.nw_type == "LJ_Harmonic" or self.nw_type == "LJ_Harmonic_simple" or self.nw_type == "Atr_LJ_w_Ang":
			multi_nw = round(target_density/init_dens)
			density = multi_nw*n_beads_unit/a_cell**3						# 密度
			err_dens = (density/target_density - 1)*100
			system_size = a_cell*self.n_cell							# システムサイズ
			print("設定密度:\t\t\t", target_density)
			print("多重度:\t\t\t\t", multi_nw)
			print("システムサイズ:\t\t\t", round(system_size, 4))
			print("密度:\t\t\t\t", round(density, 4))
			print("密度の誤差:\t\t\t", round(err_dens, 3), "%")
			if abs(err_dens) > 1:
				print(u"\n##### \n圧縮後の密度が、設定したい密度と 1% 以上違います。\nそれでも計算しますか？")
				self.prompt()
			else:
				print()
				self.prompt()
		elif self.nw_type == "NoLJ_Harmonic":
			multi_nw = self.multi_phantom
			#
			density = multi_nw*n_beads_unit/a_cell**3					# 密度
			system_size = a_cell*self.n_cell						# システムサイズ
			print("多重度:\t\t\t\t", multi_nw)
			print("システムサイズ:\t\t\t", round(system_size, 4))
			print("密度:\t\t\t\t", round(density, 4))
			#
			self.prompt()
		else:
			multi_nw = 1
		print("#########################################")
		
		return multi_nw, density, system_size, a_cell

	###############
	# 計算条件の確認
	def prompt(self):
		dic={'y':True,'yes':True,'n':False,'no':False}
		print()
		while True:
			inp = input(u'計算を続行 ==> [Y]es >> ').lower()
			if inp in dic:
				inp = dic[inp]
				break
			print(u'##### \nもう一度入力してください。')
		if inp :
			print(u"計算を続行します")
		else:
			sys.exit(u"##### \n計算を中止します。")

		return

	###########################################
	# 基本構造として、8Chainモデルを設定
	def make_8chain_dic(self):
		# システム全体にわたるピボットのxyzとIDの辞書を作成
		jp_id_dic, jp_xyz_dic, atom_jp, n_jp = self.set_jp_id()
		# ストランドの結合状態を記述
		init_8ch_dic, vector_dic = self.set_strands(jp_id_dic)
		return init_8ch_dic, jp_xyz_dic, atom_jp, n_jp, vector_dic 

	##########################################
	# システム全体にわたるjpのxyzとIDの辞書を作成
	def set_jp_id(self):
		jp_id = 0
		jp_id_dic = {}
		jp_xyz_dic = {}
		atom_jp = []
		for z in range(n_cell):
			for y in range(n_cell):
				for x in range(n_cell):
					base_xyz = np.array([x,y,z])
					for jp in self.jp_xyz:
						jp_id_dic[tuple(np.array(jp) + base_xyz)] = (jp_id)
						jp_xyz_dic[jp_id] = tuple(np.array(jp) + base_xyz)
						atom_jp.append([jp_id, 0, 0])
						jp_id += 1
		n_jp = jp_id
		return jp_id_dic, jp_xyz_dic, atom_jp, n_jp

	#####################################################
	# ストランドの結合状態を記述
	def set_strands(self, jp_id_dic):
		init_8ch_dic = {}
		vector_dic = {}
		str_id = 0
		for z in range(n_cell):
			for y in range(n_cell):
				for x in range(n_cell):
					base_xyz = np.array([x,y,z])
					for xyz in self.strand_se_xyz:
						# サブチェインの末端間のベクトルを設定
						start_xyz = np.array(self.start_jp) + base_xyz
						end_xyz = np.array(xyz[1]) + base_xyz
						vector = end_xyz - start_xyz
						# 始点のアトムのIDを設定
						start_id = jp_id_dic[tuple(start_xyz)]
						# 終点のアトムのIDを周期境界条件で変更
						mod_end_xyz = list(end_xyz)[:]
						for i in range(3):
							if mod_end_xyz[i] == n_cell:
								mod_end_xyz[i] = 0
						end_id = jp_id_dic[tuple(mod_end_xyz)]
						#
						init_8ch_dic[str_id] = tuple([start_id, end_id])
						vector_dic[str_id] = vector
						str_id+=1
		return init_8ch_dic, vector_dic


################################################################################
## トポロジーの異なるネットワークを探索して、代数的連結性の分布関数を策定
################################################################################
class ModifyTop:
	def __init__(self, init_8ch_dic, n_chains, n_jp, n_cell, pre_try, pre_sampling, n_try, n_sampling, f_rand, f_pool, multi_nw, hist_bins):
		self.init_dic = init_8ch_dic
		self.n_chains = n_chains
		self.n_jp = n_jp
		self.n_cell = n_cell
		self.pre_try = pre_try 
		self.pre_sampling = pre_sampling
		self.n_try = n_try
		self.n_sampling = n_sampling
		self.f_rand = f_rand
		self.f_pool = f_pool
		self.multi_nw = multi_nw
		self.hist_bins = hist_bins

	#########################################################
	# トポロジーの異なるネットワークを探索して、代数的連結性の分布関数を策定し、ネットワークトポロジーの配列辞書を決める。
	def top_search(self):
		target_dir = str(self.n_chains) +"_chains_" + str(self.n_cell) + "_cells_"
		target_dir += str(self.n_try) + "_trials_" + str(self.n_sampling) + "_sampling"
		os.makedirs(target_dir, exist_ok = True)
		#
		candidate_list = []
		if self.f_rand == 1: # フラグが立っているときはつなぎ替える
			candidate_list = self.strand_exchange()
			#
			with open(os.path.join(target_dir, 'init.pickle'), mode = 'wb') as f:
				pickle.dump(candidate_list, f)
			#
			print("##################################################")
			print("Trial, Sampling = ", self.n_try, ", ", self.n_sampling)
			print("Total Sampling = ", self.n_try*self.n_sampling)
			print("Initial Candidates = ", len(candidate_list))
			print("##################################################")
			# ヒストグラム中の最大頻度を与えるネットワークトポロジーの配列辞書を決める。
			top_dic_list = self.nw_search(candidate_list, target_dir)
			#
			return top_dic_list
		elif self.f_rand == 0:
			count = 0
			while count < self.n_sampling:
				print("\rSampling: {0} in {1}".format(count, self.n_sampling), end = '')
				reduced_nw_dic, alg_const = self.random_reduce()
				candidate_list.append([alg_const, reduced_nw_dic])
				count += 1
			print()
			# 
			self.make_histgram(candidate_list, target_dir)
			sys.exit()
	
	#########################################################
	# 過去の探索データを使って、代数的連結性の分布関数を選択
	def top_select(self, read_file_path):
		with open(os.path.join(read_file_path, 'init.pickle'), mode = 'rb') as f:
			candidate_list = pickle.load(f)
		print("##################################################")
		print("Reloaded Candidates = ", len(candidate_list))
		print("##################################################")
		# ヒストグラム中の最大頻度を与えるネットワークトポロジーの配列辞書を決める。
		top_dic_list = self.nw_search(candidate_list, read_file_path)
		return top_dic_list

	#####################################################
	# 任意のストランドを選択し、ストランドの繋ぎ変えを行う
	def strand_exchange(self):
		tmp_list = []
		final_list = []
		# p = Pool(multiprocessing.cpu_count() - 4)
		p = Pool(self.f_pool)
		result = p.map(self.pre_search, range(self.pre_sampling))
		for i in result:
			tmp_list.extend(i)
		print("##################################################")
		print("Pre_Search Result:", len(tmp_list))
		print("##################################################")
		#
		for i in range(self.n_sampling):
			final_list.extend(self.search_second(tmp_list, i))

		return final_list

	#########################################################
	# 任意のストランドを選択し、所望の分岐数にストランドを消去
	def random_reduce(self):
		alg_const_init = self.calc_lap_mat(self.init_dic)
		flag = 1
		while flag == 1:
			tmp_dic = copy.deepcopy(self.init_dic)
			del_bond = []
			# 消去対象のストランドをリストアップ
			# ユニットセル中のストランドから必要な数だけ抽出
			tmp_del = random.sample(range(8), (8 - n_chains))
			# 全セルに渡って、消去するストランドのリストを作成
			for i in range(self.n_cell**3):
				del_bond.extend(list(8*i + np.array(tmp_del)))
			# ストランドを消去した辞書とその代数的連結性を得る。
			for target in del_bond:
				deleted = tmp_dic.pop(target)
			alg_const = self.calc_lap_mat(tmp_dic)
			#
			if alg_const_init > alg_const:
				flag = 0

		return tmp_dic, alg_const

	########################################################
	# ストランドの繋ぎ変えを行う
	def pre_search(self, x):
		dic, alg_const = self.random_reduce()
		print("pre_sampling ID =", x,  "Initial Algebratic Conectivity =", alg_const)
		#
		tmp_list = []
		count = 0
		failed = 0
		show = 5
		while count < self.pre_try:
			# 現状のストランドのリストの中からランダムに一つ選択し、"selected_strand"とする。
			selected_strand = random.choice(list(dic.keys()))
			# 繋ぎ変え得るストランドのリストを現状のネットワークと比較して、交換可能なセットを見つける。
			tmp_dic, alg_const = self.find_pair(selected_strand, dic)
			if alg_const != 0:
				count += 1
				tmp_list.append([alg_const, tmp_dic])
				failed = 0
				if count != 0 and round(show*count/self.pre_try) == show*count//self.pre_try and round(show*count/self.pre_try) == -(-show*count//self.pre_try):
					print("pre_sampling ID =", x, "count = ", count)
			else:
				failed +=1
			if failed >= self.pre_try:
				print("##########################################")
				print("pre_sampling ID =", x,  " FAILED!! with ", failed, "th trials.")
				print("##########################################")
				count = self.pre_try
				failed = 0
		# 
		return tmp_list

	########################################################
	# ストランドの繋ぎ変えを行う
	def search_second(self, tmp_list, x):
		dic = random.choice(tmp_list)[1]
		alg_const = self.calc_lap_mat(dic)
		print("Sampling ID =", x,  "Initial Algebratic Conectivity =", alg_const)
		#
		tmp_list = []
		count = 0
		failed = 0
		show = 5
		while count < self.n_try:
			# 現状のストランドのリストの中からランダムに一つ選択し、"selected_strand"とする。
			selected_strand = random.choice(list(dic.keys()))
			# 繋ぎ変え得るストランドのリストを現状のネットワークと比較して、交換可能なセットを見つける。
			tmp_dic, alg_const = self.find_pair(selected_strand, dic)
			if alg_const != 0:
				count += 1
				tmp_list.append([alg_const, tmp_dic])
				failed = 0
				if count != 0 and round(show*count/self.n_try) == show*count//self.n_try and round(show*count/self.n_try) == -(-show*count//self.n_try):
					print("Sampling ID =", x, "count = ", count)
			else:
				failed +=1
			if failed >= self.n_try:
				print("##########################################")
				print("Sampling ID =", x,  " FAILED!! with ", failed, "th trials.")
				print("##########################################")
				count = self.n_try
				failed = 0
		# 
		return tmp_list

	################################################################
	# 交換可能なストランドのペアを見つける
	def find_pair(self, selected_strand, dic):
		deleted_dic = dict(self.init_dic.items() - dic.items())
		# 選択したストランドの両端のjp（"jp_pairs"）を見つけ、
		# それと繋がり得る可能性のあるjpをリストアップ（"connected_list"）
		# 繋ぎ変え得るストランドを見つける（"possible_strand"）
		possible_strand = self.get_connected_jp(self.init_dic, selected_strand)
		# 繋ぎ変え得るストランドのリストを現状のネットワークと比較して、交換可能なセットを見つける。
		found_dic = {}
		for p_str in possible_strand:
			if p_str in dic:
				# 現行のストランドリストの中にいないものを選択
				count = 0
				tmp_dic = {}
				for tt in dic[selected_strand]:
					for uu in dic[p_str]:
						tmp_str = []
						if tt != uu:
							if (tt,uu) in deleted_dic.values():
								tmp_str = [k for k, v in self.init_dic.items() if v == (tt, uu)]
							elif (uu,tt) in deleted_dic.values():
								tmp_str = [k for k, v in self.init_dic.items() if v == (uu, tt)]
							if tmp_str != []:
								tmp_dic[tmp_str[0]] = tuple(self.init_dic[tmp_str[0]])
								count +=1
				if count == 2 and tmp_dic != {}:
					found_dic.update(tmp_dic)
					dic.pop(p_str)
					dic.pop(selected_strand)
					dic.update(tmp_dic)
					alg_const = self.calc_lap_mat(dic)
					return dic, alg_const
				else:
					alg_const = 0
		return dic, alg_const

	#################################################################
	# 任意のjpと連結したjpを見つけ、繋がり得る可能性のあるjpをリストアップ
	def get_connected_jp(self, dic, selected_strand):
		# 選択したストランドの両端のjp（"jp_pairs"）を見つけ、
		jp_pairs = tuple(dic[selected_strand])
		# ラプラシアン行列を利用して、それと繋がり得る可能性のあるjpをリストアップ（"connected_list"）
		lap_mat = self.make_lap_mat(dic)
		connected_list = []
		for t_jp in jp_pairs:
			t_list = list(lap_mat[t_jp])
			connected_list.append(self.find_index(t_list))
		# 繋ぎ変え得るストランドを見つける。
		possible_strand = []
		for i in connected_list[0]:
			for j in connected_list[1]:
				link = self.get_link(dic, i, j)
				if link:
					# print(link)
					possible_strand.append(link[0])
		return possible_strand

	############################################################################
	# ラプラシアン行列を利用して、任意のjpとストランドによりつながる８個のjpを見つける。
	def find_index(self, t_list):
		con_list = [i for i, x in enumerate(t_list) if x == -1.0]
		return con_list

	#######################################################
	# 任意の二つのjpがストランドによりつながっている可能性を調査
	# 繋がり得るものであれば、そのストランドのID番号を返す。
	def get_link(self, dic, jp0, jp1):
		val_tpl = tuple([jp0, jp1])
		# print("val", val_tpl)
		link = [k for k, v in dic.items() if v == val_tpl]
		# print(link)
		if link:
			return link
		elif link == []:
			val_tpl = tuple([jp1, jp0])
			link = [k for k, v in dic.items() if v == val_tpl]
			if link:
				return link
		else:
			return []

	####################################################
	# 任意のネットワークトポロジーから、ラプラシアン行列を作成
	def calc_lap_mat(self, topl_dic):
		lap_mat = self.make_lap_mat(topl_dic)
		# 固有値を計算
		la, v = np.linalg.eig(lap_mat)
		# 固有値の二番目の値を見つける。
		alg_const = round(sorted(np.real(la))[1], 3)
		return alg_const

	####################################################
	# 任意のネットワークトポロジーから、ラプラシアン行列を作成
	def make_lap_mat(self, topl_dic):
		lap_mat = np.zeros((self.n_jp, self.n_jp))
		for i in topl_dic:
			lap_mat[topl_dic[i][0], topl_dic[i][1]] = -1
			lap_mat[topl_dic[i][1], topl_dic[i][0]] = -1
			lap_mat[topl_dic[i][0], topl_dic[i][0]] += 1
			lap_mat[topl_dic[i][1], topl_dic[i][1]] += 1
		return lap_mat

	#####################################################################
	# ヒストグラム中の最大頻度を与えるネットワークトポロジーの配列辞書を決める。
	def nw_search(self, candidate_list, target_dir):
		error = 1E-5
		tmp_list = []
		val_list = []
		data_list = []
		# ヒストグラムを作成
		x, val = self.make_histgram(candidate_list, target_dir)
		# 最頻値のレンジを決める
		val_range = self.find_range(x, val)
		# 上記のレンジに入る配列をピックアップ
		for i in candidate_list:
			if i[0] >= val_range[0] and i[0] <= val_range[1]:
				tmp_list.append(i)
		random.shuffle(tmp_list)
		# 
		count = 0
		for i, selected_list in enumerate(tmp_list):
			# print(i, count, selected_list[0], val_list)
			if selected_list[0] not in val_list:
				val_list.append(selected_list[0])
				data_list.append(selected_list[1])
				count += 1
			if len(val_list) == self.multi_nw:
				with open(os.path.join(target_dir, 'selected_val.dat'), 'w') as f:
					f.write("Selected arg. con.\n\n")
					for i in val_list:
						print("Selected arg. con.", round(i, 4))
						f.write("arg. con. = " + str(round(i, 4)) + '\n')
				return data_list
		#
		print("No effective list was found for multi numbers of", self.multi_nw, "!  Try again!!")
		sys.exit()

	# 最大頻度の範囲を決める。
	def find_range(self, x, val):
		index = np.where(val == max(val))[0][0]
		f_range = 0
		value = 0
		while value < self.multi_nw:
			value = 0
			f_range += 1
			for i in range(2*f_range + 1):
				if val[index - f_range + i] != 0:
					value += 1
		val_range = [ x[index - f_range], x[index + f_range] ]
		#
		print("##################################################")
		print("Most frequent range = ", round(val_range[0], 5), " to ", round(val_range[1], 5))
		key = input("OK? input y or n: ")
		if key == 'y' or key == 'Y':
			pass
		else:
			print("##################################################")
			print("Input new range:")
			low = float(input("low=: "))
			high = float(input("high=: "))
			val_range = list([low, high])
		
		return val_range

	############################################################
	#----- ヒストグラムのグラフの作成
	def make_histgram(self, candidate_list, target_dir):
		# すべての行の０列目を選択
		tmp_list = list(np.array(candidate_list)[:,0])
		# ヒストグラムを作成
		hist = np.histogram(tmp_list, self.hist_bins)
		# データを整理
		x = hist[1]
		val = hist[0]
		# グラフ用にデータを変更
		bin_width = (x[1]-x[0])
		mod_x = (x + bin_width/2)[:-1]
		hist_data = np.stack([mod_x, val], axis = 1)
		#
		self.write_data(target_dir, hist_data)
		#
		self.make_graph(target_dir, bin_width)
		return x, val
	
	# ヒストグラムのデータを書き出し 
	def write_data(self, target_dir, list):
		with open(os.path.join(target_dir, "hist.dat"), 'w') as f:
			f.write("# Histgram data:\n# Arg. Con.\tFreq.\n\n")
			for line in list:
				f.write(str(line[0]) + '\t' + str(line[1])  + '\n')

	###########################################################
	#----- グラフを作成
	def make_graph(self, target_dir, bin_width):
		self.make_script(target_dir, bin_width)
		os.chdir(target_dir)
		if platform.system() == "Windows":
			subprocess.call(["make_hist.plt"], shell=True)
		elif platform.system() == "Linux":
			subprocess.call(['gnuplot ' + "make_hist.plt"], shell=True)
		os.chdir("..")

	# 必要なスクリプトを作成
	def make_script(self, target_dir, bin_width):
		with open(os.path.join(target_dir, "make_hist.plt"), 'w') as f:
			script = self.script_content(bin_width)
			f.write(script)
		return

	# スクリプトの中身
	def script_content(self, bin_width):
		#
		script = 'set term pngcairo font "Arial,14" \nset colorsequence classic \n'
		#
		script += '# \ndata = "hist.dat" \nset output "Histgram.png"\n'
		#
		script += '# set label 1 sprintf("Tg = %.3f", tg) left at tg, y_tg-20 \n'
		#
		script += '#\nset size square\n# set xrange [0:1.0]\n#set yrange [0:100]\n'
		#
		script += '#\nset xlabel "Arg. Con."\nset ylabel "Freq."\n'
		#
		script += 'set style fill solid 0.5\nset boxwidth ' + str(bin_width) + '\n'
		#
		script += '#\nplot	data w boxes noti'
		
		return script


################################################################################
# ターゲットとなるネットワーク全体の辞書を決める。
################################################################################
class SetUp:
	def __init__(self, top_dic_list, vector_dic, atom_jp, jp_xyz_dic, n_strand, n_sc):
		self.top_dic_list = top_dic_list
		self.vector_dic = vector_dic
		self.atom_jp = atom_jp
		self.jp_xyz_dic = jp_xyz_dic
		self.n_strand = n_strand
		self.n_sc = n_sc

	def make_data_dic(self):
		calcd_data_dic_list = []
		for str_top_dic in self.top_dic_list:
			atom_all = []
			pos_all = {}
			#
			strand_xyz, bond_all, atom_strand, angle_all = self.set_strands(str_top_dic)
			atom_all.extend(self.atom_jp)
			atom_all.extend(atom_strand)
			pos_all.update(self.jp_xyz_dic)
			pos_all.update(strand_xyz)
			#
			calcd_data_dic = {"atom_all":atom_all, "bond_all":bond_all, "pos_all":pos_all, "angle_all":angle_all}
			calcd_data_dic_list.append(calcd_data_dic)
		return calcd_data_dic_list

	# 一本のストランド中の各アトムのxyzリストとボンドリストを作成
	def set_strands(self, str_top_dic):
		strand_xyz = {}
		bond_all = {}
		atom_strand = []
		angle_all = []
		seq_atom_id = len(self.jp_xyz_dic)
		bond_id = 0
		# 
		for strand_id in str_top_dic:
			# 各ストランドの始点と終点のjpごとのXYZで処理する
			start_id = str_top_dic[strand_id][0]
			end_id = str_top_dic[strand_id][1]
			vector = self.vector_dic[strand_id]
			start_xyz = np.array(self.jp_xyz_dic[start_id])
			end_xyz = np.array(self.jp_xyz_dic[end_id])
			tmp_xyz, tmp_bond, tmp_atom_st, tmp_angle, seq_atom_id, bond_id = self.calc_single_strand(start_id, end_id, vector, start_xyz, end_xyz, seq_atom_id, bond_id)
			# print(strand_id)
			# print(tmp_xyz, tmp_bond, tmp_atom_st, tmp_angle, seq_atom_id, bond_id )
			strand_xyz.update(tmp_xyz)
			bond_all.update(tmp_bond)
			atom_strand.extend(tmp_atom_st)
			angle_all.append(tmp_angle)
		return strand_xyz, bond_all, atom_strand, angle_all

	###############################################################
	# 一本のストランド中の各アトムのxyzリストとボンドリストを作成
	def calc_single_strand(self, start_id, end_id, vector, start_xyz, end_xyz, seq_atom_id, bond_id):
		tmp_xyz = {}
		tmp_bond = {}
		tmp_angle = []
		tmp_atom_st = []
		# 始点のアトムのIDを設定
		s_id = start_id
		tmp_angle.append(s_id)
		# ストランドの鎖長分のループ処理
		unit_len = 1./(self.n_strand + 1)
		ortho_vec = self.find_ortho_vec(vector)
		mod_o_vec = np.linalg.norm(vector)*ortho_vec
		for seg in range(self.n_strand):	
			#
			pos = tuple(start_xyz + vector*(seg + 1)*unit_len)
			tmp_xyz[seq_atom_id] = pos
			tmp_atom_st.append([seq_atom_id, 1, 1])
			e_id = seq_atom_id
			#
			if seg == 0:
				bond = 0
			else:
				bond = 1
			tmp_bond[bond_id] = tuple([bond, [s_id, e_id]])
			bond_id += 1
			tmp_angle.append(e_id)
			s_id = e_id
			seq_atom_id += 1
			#
			if self.n_sc != 0:
				sc_s_id = s_id
				for i in range(self.n_sc):
					tmp_xyz[seq_atom_id] = tuple(np.array(pos)  +  (i + 1)*mod_o_vec*unit_len)
					# print(tuple(np.array(pos)  +  (i + 1)*ortho_vec*unit_len))
					tmp_atom_st.append([seq_atom_id, 2, 1])
					sc_e_id = seq_atom_id
					#
					bond = 2
					tmp_bond[bond_id] = tuple([bond, [sc_s_id, sc_e_id]])
					sc_s_id = sc_e_id
					seq_atom_id += 1
					bond_id += 1
		e_id = end_id
		bond = 0
		tmp_bond[bond_id] = tuple([bond, [s_id, e_id]])
		tmp_angle.append(e_id)
		bond_id += 1

		return tmp_xyz, tmp_bond, tmp_atom_st, tmp_angle, seq_atom_id, bond_id

	#####
	def find_ortho_vec(self, list):
		vec = np.array(list).reshape(-1,1)
		# 線形独立である新たな三次元ベクトルを見つける。
		rank = 0
		while rank != 2:
			a = np.array(np.random.rand(3)).reshape(-1,1)
			target = np.hstack((vec, a))
			rank = np.linalg.matrix_rank(target)
		# QR分解により
		q, r = np.linalg.qr( target )
		# print(q[:,1])
		ortho_vec = q[:,1]
		return ortho_vec


##########################################
# Initi_UDF の作成
##########################################
class MakeBaseUDF:
	def __init__(self, calcd_data_dic_list, blank_udf, multi_nw, system_size, a_cell):
		self.calcd_data_dic_list = calcd_data_dic_list
		self.blank_udf = blank_udf
		self.multi_nw = multi_nw
		self.system_size = system_size
		self.a_cell = a_cell
		self.template = "base_uin.udf"
		# 初期状態 [R0, K]
		self.harmonic_cond = [0.967, 50]
		# [Potential_Type, theta0, K]		
		self.angle_cond = ['Theta2', 0, 0.8] 	
		# [Cutoff, Scale_1_4_Pair, sigma, epsilon, range]
		self.lj_cond = [2**(1/6), 1.0, 1.0, 1.0, 1.0]			
		# Cognac用の名称設定
		self.nw_name = "Network"
		self.atom_name = ["JP_A", "Strand_A", "Side_A"]
		self.bond_name = ["bond_JP-Chn", "bond_Chain", "bond_Side"]
		self.angle_name = ["angle_AAA"]
		self.site_name = ["site_JP", "site_Chain"]
		self.pair_name = ["site_JP-site_JP", "site_Chain-site_JP", "site_Chain-site_Chain"]
		self.site_pair_name = [ ["site_JP", "site_JP"], ["site_Chain", "site_JP"], ["site_Chain", "site_Chain"]]

	################################################################################
	# base_udfを作成
	def makeudf(self):
		# 初期udfの内容を作成する
		base_udf = self.base_udf()
		# すべてのアトムの位置座標及びボンド情報を設定
		self.setup_atoms()
		#
		return base_udf

	################################################################################
	def base_udf(self):
		#--- create an empty UDF file ---
		f = open(self.template,'w')
		f.write('\include{"%s"}' % self.blank_udf)
		f.close()

		u = UDFManager(self.template)
		# goto global data
		u.jump(-1)

		#--- Simulation_Conditions ---
		p = "Simulation_Conditions.Dynamics_Conditions.Moment."
		u.put(10000, p + "Interval_of_Calc_Moment")
		u.put(1, p + "Calc_Moment")
		u.put(1, p + "Stop_Translation")
		u.put(1, p + "Stop_Rotation")

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		p = 'Initial_Structure.Initial_Unit_Cell.'
		u.put(0, p + 'Density')
		u.put([self.system_size, self.system_size, self.system_size, 90.0, 90.0, 90.0], p + 'Cell_Size')

		#--- Molecular_Attributes ---
		# Atomes
		for i in range(len(self.atom_name)):
			p = 'Molecular_Attributes.Atom_Type[].'
			u.put(self.atom_name[i], 	p + 'Name', [i])
			u.put(1.0, 					p + 'Mass', [i])
		# Bond
		for i in range(len(self.bond_name)):
			p = 'Molecular_Attributes.Bond_Potential[].'
			u.put(self.bond_name[i], 		p + 'Name', [i])
			u.put('Harmonic', 				p + 'Potential_Type', [i])
			u.put(self.harmonic_cond[0], 	p + 'R0', [i])
			u.put(self.harmonic_cond[1], 	p + 'Harmonic.K', [i])
		# Angle
		for i in range(len(self.angle_name)):
			p = 'Molecular_Attributes.Angle_Potential[].'
			u.put(self.angle_name[i], p + 'Name', [i])
			u.put(self.angle_cond[0], p + 'Potential_Type', [i])
			u.put(self.angle_cond[1], p + 'theta0', [i])
			u.put(self.angle_cond[2], p + 'Theta2.K', [i])

		# Site
		for i in range(len(self.site_name)):
			p = 'Molecular_Attributes.Interaction_Site_Type[].'
			u.put(self.site_name[i], 	p + 'Name', [i])
			u.put(1, 					p + 'Num_of_Atoms', [i])
			u.put(self.lj_cond[4], 		p + 'Range', [i])

		#--- Pair_Interaction[] ---
		for i in range(len(self.pair_name)):
			p = 'Interactions.Pair_Interaction[].'
			u.put(self.pair_name[i],   			p + 'Name', [i])
			u.put('Lennard_Jones', 				p + 'Potential_Type', [i])
			u.put(self.site_pair_name[i][0],	p + 'Site1_Name', [i])
			u.put(self.site_pair_name[i][1],	p + 'Site2_Name', [i])
			u.put(self.lj_cond[0],				p + 'Cutoff', [i])
			u.put(self.lj_cond[1],				p + 'Scale_1_4_Pair', [i])
			u.put(self.lj_cond[2],				p + 'Lennard_Jones.sigma', [i])
			u.put(self.lj_cond[3],				p + 'Lennard_Jones.epsilon', [i])

		#--- Write UDF ---
		u.write(self.template)

		return self.template

	################################################################################
	# すべてのアトムの位置座標及びボンド情報を設定
	def setup_atoms(self):
		# 多重配置の場合の位置シフト量
		shift = 2
		#
		u = UDFManager(self.template)
		u.jump(-1)

		#--- Set_of_Molecules の入力
		p = 'Set_of_Molecules.molecule[].'
		pa = p + 'atom[].'
		pi = p + 'interaction_Site[].'
		pb = p + 'bond[].'
		pang = p +  'angle[].'
		#
		count = 0
		for mul in range(self.multi_nw):
			atom_all = self.calcd_data_dic_list[mul]["atom_all"]
			bond_all = self.calcd_data_dic_list[mul]["bond_all"]
			pos_all = self.calcd_data_dic_list[mul]["pos_all"]
			angle_all = self.calcd_data_dic_list[mul]["angle_all"]
			#
			u.put(self.nw_name + '_' + str(mul), p + 'Mol_Name', [count])
			# beads
			n_atom = 0
			for atom in atom_all:
				# atom
				id_shift = len(atom_all)
				u.put(atom[0] + count*id_shift, pa + 'Atom_ID', [count, n_atom])
				u.put(self.atom_name[atom[1]], 	pa + 'Atom_Name', [count, n_atom])
				u.put(self.atom_name[atom[1]], 	pa + 'Atom_Type_Name', [count, n_atom])
				u.put(0, 						pa + 'Chirality', [count, n_atom])
				u.put(1, 						pa + 'Main_Chain', [count, n_atom])
				# interaction site
				u.put(self.site_name[atom[2]], 	pi + 'Type_Name', [count, n_atom])
				u.put(n_atom, 					pi + 'atom[]', [count, n_atom, 0])
				n_atom += 1
			# bonds
			n_bond = 0
			ang_list=[]
			for bond in range(len(bond_all)):
				u.put(self.bond_name[bond_all[bond][0]], 	pb + 'Potential_Name', [count, n_bond])
				u.put(bond_all[bond][1][0], 				pb + 'atom1', [count, n_bond])
				ang_list.append(bond_all[bond][1][0])
				u.put(bond_all[bond][1][1], 				pb + 'atom2', [count, n_bond])
				n_bond += 1

			# angles
			n_ang = 0
			for a_list in angle_all:
				for j in range(len(a_list)-2):
					u.put(self.angle_name[0], 	pang + 'Potential_Name', [count, n_ang])
					u.put(a_list[j], 			pang + 'atom1', [count, n_ang])
					u.put(a_list[j + 1], 		pang + 'atom2', [count, n_ang])
					u.put(a_list[j + 2], 		pang + 'atom3', [count, n_ang])
					n_ang += 1

			# Draw_Attributes
			color = ["Red", "Green", "Blue", "Magenta", "Cyan", "Yellow", "White", "Black", "Gray"]
			mm = mul % 9
			u.put([self.nw_name + '_' + str(mul), color[mm], 1.0, 1.0], 'Draw_Attributes.Molecule[]', [count])
			count += 1

		# アトムの座標位置を、シフトしながら、設定
		sp = 'Structure.Position.mol[].atom[]'
		count = 0
		#
		for mul in range(self.multi_nw):
			shift_vec = count*shift*np.array(np.random.rand(3))
			pos_all = self.calcd_data_dic_list[mul]["pos_all"]
			for i in range(len(pos_all)):
				mod_pos = self.a_cell*np.array(list(pos_all[i])) + shift_vec
				u.put(list(mod_pos), sp, [count, i])
			count+=1

		#--- Write UDF ---
		u.write(self.template)
		return


#######################################
# ファイル名を設定し、バッチファイルを作成
#######################################
class MakeBatch:
	def __init__(self, target_name, base_udf, ver_cognac, core, nw_type):
		self.target_name = target_name
		self.base_udf = base_udf
		self.ver_cognac = ver_cognac
		self.core = core
		self.nw_type = nw_type
		# self.press_comp = [0.2, 1.0]

	#----- ファイル名を設定し、バッチファイルを作成
	def setup_all(self):
		batch_all = ''
		batch = ''
		#
		if platform.system() == "Windows":
			batch += "title Calculating_Pre\n"
		elif platform.system() == "Linux":
			batch += "#!/bin/bash\n"
			batch += "echo -en '\[\e]0;'" + "Calculating" + r"_Pre'\a\]\u@\h:\w\$ '" + " \n" 
		
		############################################
		# 各ネットワーク状態に応じた計算の初期状態を設定
		if self.nw_type == "NoLJ_Harmonic" or self.nw_type == "LJ_Harmonic" or self.nw_type == "KG":
			# シミュレーション時間の設定
			time = [0.01, 10000000, 100000]
			#
			present_udf = 'Pre_' + self.target_name + "_uin.udf"
			self.setup_init_udf(self.base_udf, present_udf, time)
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			read_udf= present_udf.replace("uin", "out")
			template = present_udf
		
		##################################################
		# まったく絡み合わせないセットアップ、ボンドはハーモニック
		if self.nw_type == "LJ_Harmonic_simple":
			# シミュレーション時間の設定
			time = [0.01, 10000, 1000]
			#
			present_udf = 'Pre_' + self.target_name + "_softbond_uin.udf"
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			# 初期状態の設定
			self.setup_simple_udf(self.base_udf, present_udf, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf
			# ハーモニックポテンシャルの再設定
			time = [0.01, 10000000, 100000]
			present_udf = 'Pre_' + self.target_name + "_uin.udf"
			udf_files = [template, read_udf, present_udf]
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			self.bond_harm(udf_files, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf

		##################################################
		# 充分に絡み合わせるセットアップ、ボンドはハーモニック
		if self.nw_type == "LJ_Harmonic" or self.nw_type == "KG":
			# シミュレーション時間の設定
			time = [0.01, 1000000, 10000]
			# アングルポテンシャルの設定
			present_udf = 'Pre_' + self.target_name + "_Angle_uin.udf"
			udf_files = [template, read_udf, present_udf]
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			self.set_ang(udf_files, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf
			# 斥力ポテンシャルの設定
			time = [0.01, 5000000, 50000]
			present_udf = 'Pre_' + self.target_name + "_rep_LJ_uin.udf"
			udf_files = [template, read_udf, present_udf]
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			self.rep_lj(udf_files, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf
			# ハーモニックポテンシャルの再設定
			time = [0.01, 5000000, 50000]
			present_udf = 'Pre_' + self.target_name + "_rep_LJ_Harmonic_uin.udf"
			udf_files = [template, read_udf, present_udf]
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			self.bond_harm(udf_files, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf

		# ###################################
		# # 充分に絡み合わせるKG用のセットアップ
		# if self.nw_type == "KG":
		# 	# 
		# 	rmax_list = [1.5]
		# 	for rmax in rmax_list:
		# 		present_udf = 'Pre_' + self.target_name + "_FENE_" + str(rmax) + "_uin.udf"
		# 		udf_files = [template, read_udf, present_udf]
		# 		batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 		self.bond_fene(udf_files, rmax)
		# 		read_udf= present_udf.replace("uin", "out")
		# 		template = present_udf



		# #
		# elif self.nw_type == "Atr_LJ_w_Ang":
		# 	#
		# 	press = 0.2
		# 	present_udf = 'Pre_' + self.target_name + "_Calc_rep_LJ_uin.udf"
		# 	udf_files = [template, read_udf, present_udf]
		# 	batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 	self.mod_rep_udf(udf_files, press)
		# 	read_udf= present_udf.replace("uin", "out")
		# 	template = present_udf
		# 	#
		# 	present_udf = 'Pre_' + self.target_name + "_Calc_Bond_uin.udf"
		# 	udf_files = [template, read_udf, present_udf]
		# 	batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 	self.mod_bond_udf(udf_files)
		# 	read_udf= present_udf.replace("uin", "out")
		# 	template = present_udf
		# 	#
		# 	present_udf = 'Pre_' + self.target_name + "_Calc_atrct_LJ_uin.udf"
		# 	udf_files = [template, read_udf, present_udf]
		# 	batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 	self.mod_atrct_udf(udf_files)
		# 	read_udf= present_udf.replace("uin", "out")
		# 	template = present_udf
		# 	#
		# 	present_udf = 'Pre_' + self.target_name + "_Calc_Ang_uin.udf"
		# 	udf_files = [template, read_udf, present_udf]
		# 	batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 	self.mod_ang_udf(udf_files)
		# 	read_udf= present_udf.replace("uin", "out")
		# 	template = present_udf	
		# 	#		
		# 	for pressure in press_comp:
		# 		present_udf = 'Pre_' + self.target_name + "_Comp_" + str(pressure).replace(".","_") + "_uin.udf"
		# 		udf_files = [template, read_udf, present_udf]
		# 		batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
		# 		self.mod_comp_udf(udf_files, pressure)
		# 		read_udf= present_udf.replace("uin", "out")
		# 		template = present_udf

		##################
		# 最終の平衡化計算
		# シミュレーション時間の設定
		time = [0.01, 2000000, 10000]
		repeat = 3
		#
		for i in range(repeat):
			present_udf = 'Eq_' + str(i) + "_" + self.target_name + "_uin.udf"
			udf_files = [template, read_udf, present_udf]
			if platform.system() == "Windows":
				batch += "title Calculating_Equiv\n"
			elif platform.system() == "Linux":
				batch += "echo -en '\[\e]0;'" + "Calculating_Equiv" + r"'\a\]\u@\h:\w\$ '" + " \n" 
			batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
			self.eq_udf(udf_files, time)
			read_udf= present_udf.replace("uin", "out")
			template = present_udf
		#
		if os.path.isfile("./Read_NW_Strand_E2E.py"):
			batch += "python Read_NW_Strand_E2E.py " + present_udf.replace("uin", "out")

		################################################
		# Temp Scan
		if self.nw_type == "Atr_LJ_w_Ang":
			# 温度条件
			temp_start = 1.0
			temp_end = 0.8
			cool_step = 0.01
			temp = temp_start
			# 冷却速度 
			cool_rate = 1e-5
			# 測定インターバル
			measure_step = 0.01
			#########################
			dt = 0.01
			time_cooling = [ dt, round(cool_step/dt/cool_rate), round(cool_step/dt/cool_rate/2) ]
			time_measure = [ dt, 20000, 100 ]
			#
			while temp >= temp_end:
				temp_str = '{:.2f}'.format(temp)
				temp = float(temp_str)
				if platform.system() == "Windows":
					batch += "title Calculating_T_" + temp_str.replace('.', '_') +'\n'
				elif platform.system() == "Linux":
					batch += "echo -en '\[\e]0;'" + "Calculating_T_" + temp_str.replace('.', '_') + r"'\a\]\u@\h:\w\$ '" + " \n"
				#
				present_udf = 'Cooling_' + self.target_name + '_Temp_' +  temp_str.replace('.', '_')   + "_uin.udf"
				udf_files = [template, read_udf, present_udf]
				batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
				self.mod_temp_udf(udf_files, temp, time_cooling)
				read_udf= present_udf.replace("uin", "out")
				template = present_udf
				#
				if temp_str in ['{:.2f}'.format(measure_step*i) for i in range(int(temp_start/measure_step) + 1)]:
					present_udf = 'Measure_' + self.target_name + '_Temp_' + temp_str.replace('.', '_') + "_uin.udf"
					udf_files = [template, read_udf, present_udf]
					batch += self.ver_cognac + ' -I ' + present_udf + ' -O ' + present_udf.replace("uin", "out") + ' -n ' + str(self.core) +' \n'
					self.mod_temp_udf(udf_files, temp, time_measure)
				#
				temp_str = '{:.2f}'.format(temp-cool_step)
				temp = float(temp_str)

			#
			if os.path.exists("Read_Vol.py"):
				batch += "python Read_Vol.py \n"
		
		# バッチファイルを作成
		with open('_Calc_all.bat','w') as f:
			# f.write(batch_all)
			f.write(batch)

		return

################################################################################
	# 初期状態を設定
	def setup_init_udf(self, template, present_udf, time):
		u = UDFManager(template)
		# goto global data
		u.jump(-1)
		
		#--- Simulation_Conditions ---
		# Solver
		p = 'Simulation_Conditions.Solver.'
		u.put('Dynamics', p + 'Solver_Type')
		u.put('NVT_Kremer_Grest', p + 'Dynamics.Dynamics_Algorithm')
		u.put(0.5, p + 'Dynamics.NVT_Kremer_Grest.Friction')

		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(100000000., p + 'Max_Force')
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')
		u.put(1.0, p + 'Temperature.Temperature')
		u.put(0., p + 'Pressure_Stress.Pressure')

		# Boundary_Conditions
		p = 'Simulation_Conditions.Boundary_Conditions'
		u.put(['PERIODIC', 'PERIODIC', 'PERIODIC', 1], p)

		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(1, p + 'Bond')
		u.put(0, p + 'Angle')
		u.put(0, p + 'Non_Bonding_Interchain')
		u.put(0, p + 'Non_Bonding_1_3')
		u.put(0, p + 'Non_Bonding_1_4')
		u.put(0, p + 'Non_Bonding_Intrachain')

		# Output_Flags.Statistics
		p = 'Simulation_Conditions.Output_Flags.Statistics.'
		u.put(1, p + 'Energy')
		u.put(1, p + 'Temperature')
		u.put(1, p + 'Pressure')
		u.put(0, p + 'Stress')
		u.put(0, p + 'Volume')
		u.put(1, p + 'Density')
		u.put(1, p + 'Cell')
		u.put(0, p + 'Wall_Pressure')
		u.put(0, p + 'Energy_Flow')

		# Output_Flags.Structure
		p = 'Simulation_Conditions.Output_Flags.Structure.'
		u.put(1, p + 'Position')
		u.put(0, p + 'Velocity')
		u.put(0, p + 'Force')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		if self.nw_type == "LJ_Harmonic" or self.nw_type == "LJ_Harmonic_simple" or self.nw_type == "KG":
			p = 'Initial_Structure.Initial_Unit_Cell.'
			u.put(0.85, p + 'Density')
			u.put([0, 0, 0, 90.0, 90.0, 90.0], p + 'Cell_Size')
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put(['', -1, 0, 0], 	p + 'Restart')

		# Relaxation
		p = 'Initial_Structure.Relaxation.'
		u.put(1, p + 'Relaxation')
		u.put('DYNAMICS', p + 'Method')
		u.put(300, p + 'Max_Relax_Force')
		u.put(10000, p + 'Max_Relax_Steps')

		#--- Write UDF ---
		u.write(present_udf)

		return

################################################################################
	# 絡み合いのない初期状態を設定
	def setup_simple_udf(self, template, present_udf, time):
		u = UDFManager(template)
		# goto global data
		u.jump(-1)
		
		#--- Simulation_Conditions ---
		# Solver
		p = 'Simulation_Conditions.Solver.'
		u.put('Dynamics', p + 'Solver_Type')
		u.put('NVT_Kremer_Grest', p + 'Dynamics.Dynamics_Algorithm')
		u.put(0.5, p + 'Dynamics.NVT_Kremer_Grest.Friction')

		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(100000000., p + 'Max_Force')
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')
		u.put(1.0, p + 'Temperature.Temperature')
		u.put(0., p + 'Pressure_Stress.Pressure')

		# Boundary_Conditions
		p = 'Simulation_Conditions.Boundary_Conditions'
		u.put(['PERIODIC', 'PERIODIC', 'PERIODIC', 1], p)

		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(1, p + 'Bond')
		u.put(0, p + 'Angle')
		u.put(1, p + 'Non_Bonding_Interchain')
		u.put(1, p + 'Non_Bonding_1_3')
		u.put(1, p + 'Non_Bonding_1_4')
		u.put(1, p + 'Non_Bonding_Intrachain')

		# Output_Flags.Statistics
		p = 'Simulation_Conditions.Output_Flags.Statistics.'
		u.put(1, p + 'Energy')
		u.put(1, p + 'Temperature')
		u.put(1, p + 'Pressure')
		u.put(0, p + 'Stress')
		u.put(0, p + 'Volume')
		u.put(1, p + 'Density')
		u.put(1, p + 'Cell')
		u.put(0, p + 'Wall_Pressure')
		u.put(0, p + 'Energy_Flow')

		# Output_Flags.Structure
		p = 'Simulation_Conditions.Output_Flags.Structure.'
		u.put(1, p + 'Position')
		u.put(0, p + 'Velocity')
		u.put(0, p + 'Force')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		if self.nw_type == "LJ_Harmonic" or self.nw_type == "LJ_Harmonic_simple" or self.nw_type == "KG":
			p = 'Initial_Structure.Initial_Unit_Cell.'
			u.put(0.85, p + 'Density')
			u.put([0, 0, 0, 90.0, 90.0, 90.0], p + 'Cell_Size')
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put(['', -1, 0, 0], 	p + 'Restart')

		# Relaxation
		p = 'Initial_Structure.Relaxation.'
		u.put(1, p + 'Relaxation')
		u.put('DYNAMICS', p + 'Method')
		u.put(300, p + 'Max_Relax_Force')
		u.put(10000, p + 'Max_Relax_Steps')

		#--- Simulation_Conditions ---
		# Bond
		bond_name = u.get('Molecular_Attributes.Bond_Potential[].Name')
		harmonic_cond = [0.967, 200]	
		for i in range(len(bond_name)):
			p = 'Molecular_Attributes.Bond_Potential[].'
			u.put(bond_name[i], p + 'Name', [i])
			u.put('Harmonic', p + 'Potential_Type', [i])
			u.put(harmonic_cond[0], p + 'R0', [i])
			u.put(harmonic_cond[1], p + 'Harmonic.K', [i])

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def set_ang(self, udf_files, time):
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')

		#--- Simulation_Conditions ---
		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(1, p + 'Angle')

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put([read_udf, -1, 1, 0], 	p + 'Restart')

		#--- Write UDF ---
		u.write(present_udf)

		return

	###########################
	# 斥力的な非結合相互作用を設定
	def rep_lj(self, udf_files, time):
		#
		# time = [0.01, 10000, 1000]
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')

		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(0, p + 'Angle')
		u.put(1, p + 'Non_Bonding_Interchain')
		u.put(1, p + 'Non_Bonding_1_3')
		u.put(1, p + 'Non_Bonding_1_4')
		u.put(1, p + 'Non_Bonding_Intrachain')

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put([read_udf, -1, 1, 0], 	p + 'Restart')

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def bond_harm(self, udf_files, time):
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p + 'Method')
		u.put([read_udf, -1, 1, 0], p + 'Restart')

		#--- Simulation_Conditions ---
		# Bond
		bond_name = u.get('Molecular_Attributes.Bond_Potential[].Name')
		harmonic_cond = [0.967, 1111]	
		for i in range(len(bond_name)):
			p = 'Molecular_Attributes.Bond_Potential[].'
			u.put(bond_name[i], p + 'Name', [i])
			u.put('Harmonic', p + 'Potential_Type', [i])
			u.put(harmonic_cond[0], p + 'R0', [i])
			u.put(harmonic_cond[1], p + 'Harmonic.K', [i])

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def bond_fene(self, udf_files, rmax):
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p + 'Method')
		u.put([read_udf, -1, 1, 0], p + 'Restart')
		#--- Simulation_Conditions ---
		# Bond
		bond_name = u.get('Molecular_Attributes.Bond_Potential[].Name')
		for i, b_name in enumerate(bond_name):
			p = 'Molecular_Attributes.Bond_Potential[].'
			u.put(b_name, 		p + 'Name', [i])
			u.put('FENE_LJ', 	p + 'Potential_Type', [i])
			u.put(1.0,			p + 'R0', [i])
			u.put(rmax,			p + 'FENE_LJ.R_max', [i])
			u.put(30,			p + 'FENE_LJ.K', [i])
			u.put(1.0,			p + 'FENE_LJ.sigma', [i])
			u.put(1.0,			p + 'FENE_LJ.epsilon', [i])

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def eq_udf(self, udf_files, time):
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		p = "Simulation_Conditions.Dynamics_Conditions.Moment."
		u.put(100, p + "Interval_of_Calc_Moment")
		u.put(1, p + "Calc_Moment")
		u.put(1, p + "Stop_Translation")
		u.put(1, p + "Stop_Rotation")

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0],  p+'Time.delta_T')
		u.put(time[1],  p+'Time.Total_Steps')
		u.put(time[2],  p+'Time.Output_Interval_Steps')

		# Moment
		p = "Simulation_Conditions.Dynamics_Conditions.Moment."
		u.put(0, p + "Interval_of_Calc_Moment")
		u.put(0, p + "Calc_Moment")
		u.put(0, p + "Stop_Translation")
		u.put(0, p + "Stop_Rotation")

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p+'Method')
		u.put([read_udf, -1, 1, 0], p+'Restart')
		p = 'Initial_Structure.Relaxation.'
		u.put(0, p + 'Relaxation')

		#--- Write UDF ---
		u.write(present_udf)
		return

















	
	############################
	# 斥力的な非結合相互作用を設定
	def mod_rep_udf(self, udf_files, press):
		#
		time = [0.01, 100000, 1000]
		# [Cutoff, Scale_1_4_Pair, sigma, epsilon, range]
		lj_cond = [2**(1/6), 1.0, 1.0, 1.0, 1.0]
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		site_name = ["site_JP", "site_Chain"]
		pair_name = ["site_JP-site_JP", "site_Chain-site_JP", "site_Chain-site_Chain"]
		site_pair_name = [ ["site_JP", "site_JP"], ["site_Chain", "site_JP"], ["site_Chain", "site_Chain"]]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)
		cell_mass =u.size('Set_of_Molecules.molecule[].atom[]')

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')
		u.put(press,  p + 'Pressure_Stress.Pressure')
		# Solver
		p = 'Simulation_Conditions.Solver.'
		u.put('Dynamics',	p + 'Solver_Type')
		u.put('NPT_Andersen_Kremer_Grest',	p + 'Dynamics.Dynamics_Algorithm')
		u.put(cell_mass, 	p + 'Dynamics.NPT_Andersen_Kremer_Grest.Cell_Mass')
		u.put(0.5,	p + 'Dynamics.NPT_Andersen_Kremer_Grest.Friction')
		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(1, p + 'Non_Bonding_Interchain')
		u.put(1, p + 'Non_Bonding_1_3')
		u.put(1, p + 'Non_Bonding_1_4')
		u.put(1, p + 'Non_Bonding_Intrachain')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		p = 'Initial_Structure.Initial_Unit_Cell.'
		u.put(0., p + 'Density')
		u.put([0, 0, 0, 90.0, 90.0, 90.0], p + 'Cell_Size')
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put([read_udf, -1, 1, 0], 	p + 'Restart')

		# Site
		for i in range(len(site_name)):
			p = 'Molecular_Attributes.Interaction_Site_Type[' + str(i) + '].'
			u.put(site_name[i], p + 'Name')
			u.put(1, p + 'Num_of_Atoms')
			u.put(lj_cond[4], p + 'Range')

		#--- Pair_Interaction[] ---
		for i in range(len(pair_name)):
			p = 'Interactions.Pair_Interaction[' + str(i) + '].'
			u.put(pair_name[i],   	p + 'Name')
			u.put('Lennard_Jones', 	p + 'Potential_Type')
			u.put(site_pair_name[i][0],	p + 'Site1_Name')
			u.put(site_pair_name[i][1],	p + 'Site2_Name')
			u.put(lj_cond[0],		p + 'Cutoff')
			u.put(lj_cond[1],		p + 'Scale_1_4_Pair')
			u.put(lj_cond[2],		p + 'Lennard_Jones.sigma')
			u.put(lj_cond[3],		p + 'Lennard_Jones.epsilon')

		#--- Write UDF ---
		u.write(present_udf)

		return

	####################################
	# 
	def mod_atrct_udf(self, udf_files):
		#
		time = [0.01, 1000000, 10000]
		# [Cutoff, Scale_1_4_Pair, sigma, epsilon, range]
		lj_cond = [2.5, 1.0, 1.0, 1.0, 2.0]
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		site_name = ["site_JP", "site_Chain"]
		pair_name = ["site_JP-site_JP", "site_Chain-site_JP", "site_Chain-site_Chain"]
		site_pair_name = [ ["site_JP", "site_JP"], ["site_Chain", "site_JP"], ["site_Chain", "site_Chain"]]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)
		
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(100000000., p + 'Max_Force')
		u.put(time[0], p + 'Time.delta_T')
		u.put(time[1], p + 'Time.Total_Steps')
		u.put(time[2], p + 'Time.Output_Interval_Steps')
		# Calc_Potential_Flags
		p = 'Simulation_Conditions.Calc_Potential_Flags.'
		u.put(1, p + 'Non_Bonding_Interchain')
		u.put(1, p + 'Non_Bonding_1_3')
		u.put(1, p + 'Non_Bonding_1_4')
		u.put(1, p + 'Non_Bonding_Intrachain')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		p = 'Initial_Structure.Initial_Unit_Cell.'
		u.put(0., p + 'Density')
		u.put([0, 0, 0, 90.0, 90.0, 90.0], p + 'Cell_Size')
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', 		p + 'Method')
		u.put([read_udf, -1, 1, 0], 	p + 'Restart')

		# Site
		for i in range(len(site_name)):
			p = 'Molecular_Attributes.Interaction_Site_Type[' + str(i) + '].'
			u.put(site_name[i], p + 'Name')
			u.put(1, p + 'Num_of_Atoms')
			u.put(lj_cond[4], p + 'Range')

		#--- Pair_Interaction[] ---
		for i in range(len(pair_name)):
			p = 'Interactions.Pair_Interaction[' + str(i) + '].'
			u.put(pair_name[i],   	p + 'Name')
			u.put('Lennard_Jones', 	p + 'Potential_Type')
			u.put(site_pair_name[i][0],	p + 'Site1_Name')
			u.put(site_pair_name[i][1],	p + 'Site2_Name')
			u.put(lj_cond[0],		p + 'Cutoff')
			u.put(lj_cond[1],		p + 'Scale_1_4_Pair')
			u.put(lj_cond[2],		p + 'Lennard_Jones.sigma')
			u.put(lj_cond[3],		p + 'Lennard_Jones.epsilon')

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def mod_bond_fene_udf(self, udf_files, rmax):
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p + 'Method')
		u.put([read_udf, -1, 1, 0], p + 'Restart')
		#--- Simulation_Conditions ---
		# Bond
		bond_name = ["bond_JP-Chn", "bond_Chain", "bond_Side"]
		for i, b_name in enumerate(bond_name):
			p = 'Molecular_Attributes.Bond_Potential[' + str(i) + '].'
			u.put(b_name, 	p + 'Name')
			u.put('FENE_LJ', 	p + 'Potential_Type')
			u.put(1.0,	p + 'R0')
			u.put(rmax,	p + 'FENE_LJ.R_max')
			u.put(30,	p + 'FENE_LJ.K')
			u.put(1.0,	p + 'FENE_LJ.sigma')
			u.put(1.0,	p + 'FENE_LJ.epsilon')

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def mod_bond_udf(self, udf_files):
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Initial_Structure ---
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p + 'Method')
		u.put([read_udf, -1, 1, 0], p + 'Restart')
		#--- Simulation_Conditions ---
		# Bond
		bond_name = ["bond_JP-Chn", "bond_Chain", "bond_Side"]
		if self.nw_type == "Atr_LJ_w_Ang":
			harmonic_cond = [0.967, 1111]	
		elif self.nw_type == "KG":
			harmonic_cond = [1.0, 1000]
		elif self.nw_type == "NoLJ_Harmonic":
			harmonic_cond = [1.0, 100]
		for i in range(len(bond_name)):
			p = 'Molecular_Attributes.Bond_Potential[' + str(i) +'].'
			u.put(bond_name[i], p + 'Name')
			u.put('Harmonic', p + 'Potential_Type')
			u.put(harmonic_cond[0], p + 'R0')
			u.put(harmonic_cond[1], p + 'Harmonic.K')

		#--- Write UDF ---
		u.write(present_udf)

		return

	################################################################################
	def mod_comp_udf(self, udf_files, pressure):
		#
		time_comp = [0.01, 200000, 1000]
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)
		#
		cell_mass =u.size('Set_of_Molecules.molecule[].atom[]')

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time_comp[0],  p + 'Time.delta_T')
		u.put(time_comp[1],  p + 'Time.Total_Steps')
		u.put(time_comp[2],  p + 'Time.Output_Interval_Steps')
		u.put(pressure,  p + 'Pressure_Stress.Pressure')
		# Solver
		p = 'Simulation_Conditions.Solver.'
		u.put('Dynamics',	p + 'Solver_Type')
		u.put('NPT_Andersen_Kremer_Grest',	p + 'Dynamics.Dynamics_Algorithm')
		u.put(cell_mass, 	p + 'Dynamics.NPT_Andersen_Kremer_Grest.Cell_Mass')
		u.put(0.5,	p + 'Dynamics.NPT_Andersen_Kremer_Grest.Friction')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		p = 'Initial_Structure.Initial_Unit_Cell.'
		u.put(0., p + 'Density')
		u.put([0, 0, 0, 90.0, 90.0, 90.0], p+'Cell_Size')
		# # Read_Set_of_Molecules
		# p = 'Initial_Structure.Read_Set_of_Molecules'
		# u.put([read_udf, -1], p)
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p+'Method')
		u.put([read_udf, -1, 1, 0], p+'Restart')

		#--- Write UDF ---
		u.write(present_udf)
			#
		return
	
	################################################################################
	def mod_nvt_udf(self, udf_files):
		# シミュレーション時間の設定
		time = [0.01, 100000, 1000]
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0],  p+'Time.delta_T')
		u.put(time[1],  p+'Time.Total_Steps')
		u.put(time[2],  p+'Time.Output_Interval_Steps')
		u.put(1.0, p + 'Temperature.Temperature')
		u.put(0., p + 'Pressure_Stress.Pressure')
		# Solver
		p = 'Simulation_Conditions.Solver.'
		u.put('Dynamics', p + 'Solver_Type')
		u.put('NVT_Kremer_Grest', p + 'Dynamics.Dynamics_Algorithm')
		u.put(0.5, p + 'Dynamics.NVT_Kremer_Grest.Friction')

		#--- Initial_Structure ---
		# Initial_Unit_Cell
		p = 'Initial_Structure.Initial_Unit_Cell.'
		u.put(0.85, p + 'Density')
		u.put([0, 0, 0, 90.0, 90.0, 90.0], p+'Cell_Size')
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p+'Method')
		u.put([read_udf, -1, 0, 0], p+'Restart')
		p = 'Initial_Structure.Relaxation.'
		u.put(1, p + 'Relaxation')

		#--- Write UDF ---
		u.write(present_udf)
		return



	# ################################################################################
	# def mod_eq_udf(udf_files, pressure):
	# 	#
	# 	template = udf_files[0]
	# 	read_udf = udf_files[1]
	# 	present_udf = udf_files[2]
	# 	#
	# 	u = UDFManager(template)
	# 	u.eraseRecord(0,-1)
	# 	# goto global data
	# 	u.jump(-1)

	# 	#--- Simulation_Conditions ---
	# 	# Dynamics_Conditions
	# 	p = 'Simulation_Conditions.Dynamics_Conditions.'
	# 	u.put(time[0],  p+'Time.delta_T')
	# 	u.put(time[1],  p+'Time.Total_Steps')
	# 	u.put(time[2],  p+'Time.Output_Interval_Steps')
	# 	u.put(pressure,  p+'Pressure_Stress.Pressure')
	# 	#
	# 	if ANGLE_K != 1:
	# 		u.put(1, 'Simulation_Conditions.Calc_Potential_Flags.Angle')

	# 	# Solver
	# 	p = 'Simulation_Conditions.Solver.'
	# 	u.put('Dynamics',	p+'Solver_Type')
	# 	u.put('NPT_Andersen_Kremer_Grest',	p+'Dynamics.Dynamics_Algorithm')
	# 	u.put(TOTAL_ATOM, 	p+'Dynamics.NPT_Andersen_Kremer_Grest.Cell_Mass')
	# 	u.put(0.5,	p+'Dynamics.NPT_Andersen_Kremer_Grest.Friction')
	# 	#
	# 	#--- Initial_Structure ---
	# 	# Initial_Unit_Cell
	# 	p = 'Initial_Structure.Initial_Unit_Cell.'
	# 	u.put([0, 0, 0, 90.0, 90.0, 90.0], p+'Cell_Size')
	# 	# Read_Set_of_Molecules
	# 	p = 'Initial_Structure.Read_Set_of_Molecules'
	# 	u.put([read_udf, -1], p)
	# 	# Generate_Method
	# 	p = 'Initial_Structure.Generate_Method.'
	# 	u.put('Restart', p+'Method')
	# 	u.put([read_udf, -1, 1, 0], p+'Restart')

	# 	#--- Simulation_Conditions ---
	# 	# Bond
	# 	for i in range(len(BOND_NAME)):
	# 		p = 'Molecular_Attributes.Bond_Potential[' + str(i) + '].'
	# 		u.put(BOND_NAME[i], 	p + 'Name')
	# 		if BOND == "FENE":
	# 			u.put('FENE_LJ', 	p + 'Potential_Type')
	# 			u.put(FENE_LJ[0],	p + 'R0')
	# 			u.put(FENE_LJ[1],	p + 'FENE_LJ.R_max')
	# 			u.put(FENE_LJ[2],	p + 'FENE_LJ.K')
	# 			u.put(FENE_LJ[3],	p + 'FENE_LJ.sigma')
	# 			u.put(FENE_LJ[4],	p + 'FENE_LJ.epsilon')
	# 		elif BOND == "Harm":
	# 			u.put('Harmonic', 		p + 'Potential_Type')
	# 			u.put(FIN_HARMONIC[0],	p + 'R0')
	# 			u.put(FIN_HARMONIC[1],	p + 'Harmonic.K')
		
	# 	if INTERACT == "Atract":
	# 		# Site
	# 		for i in range(len(SITE_NAME)):
	# 			p = 'Molecular_Attributes.Interaction_Site_Type[' + str(i) + '].'
	# 			u.put(SITE_NAME[i], 	p + 'Name')
	# 			u.put(1, 				p + 'Num_of_Atoms')
	# 			u.put(Atract_LJ[4], 		p + 'Range')

	# 		#--- Pair_Interaction[] ---
	# 		for i in range(len(PAIR_NAME)):
	# 			p = 'Interactions.Pair_Interaction[' + str(i) + '].'
	# 			u.put(PAIR_NAME[i],   		p + 'Name')
	# 			u.put('Lennard_Jones', 		p + 'Potential_Type')
	# 			u.put(SITE_PAIR_NAME[i][0],	p + 'Site1_Name')
	# 			u.put(SITE_PAIR_NAME[i][1],	p + 'Site2_Name')
	# 			u.put(Atract_LJ[0],			p + 'Cutoff')
	# 			u.put(Atract_LJ[1],			p + 'Scale_1_4_Pair')
	# 			u.put(Atract_LJ[2],			p + 'Lennard_Jones.sigma')
	# 			u.put(Atract_LJ[3],			p + 'Lennard_Jones.epsilon')

	# 	#--- Write UDF ---
	# 	u.write(present_udf)
	# 	return


	################################################################################
	def mod_temp_udf(self, udf_files, temp, time):
		#
		template = udf_files[0]
		read_udf = udf_files[1]
		present_udf = udf_files[2]
		#
		u = UDFManager(template)
		# goto global data
		u.jump(-1)

		#--- Simulation_Conditions ---
		# Dynamics_Conditions
		p = 'Simulation_Conditions.Dynamics_Conditions.'
		u.put(time[0],  p+'Time.delta_T')
		u.put(time[1],  p+'Time.Total_Steps')
		u.put(time[2],  p+'Time.Output_Interval_Steps')
		#
		u.put(temp, p+'Temperature.Temperature')
		#--- Initial_Structure ---
		# Read_Set_of_Molecules
		p = 'Initial_Structure.Read_Set_of_Molecules'
		u.put([read_udf, -1], p)
		# Generate_Method
		p = 'Initial_Structure.Generate_Method.'
		u.put('Restart', p+'Method')
		u.put([read_udf, -1, 1, 0], p+'Restart')
		# Relaxation
		p = 'Initial_Structure.Relaxation.'
		u.put(0, p+'Relaxation')

		#--- Write UDF ---
		u.write(present_udf)
		return


# ################################################################################
# # 必要なスクリプトを作成
# def make_script(scrpt_name):
# 	script = script_content()
# 	with open(scrpt_name, 'w') as f:
# 		f.write(script)
# 	return


# # スクリプトの中身
# def script_content():
# 	script = '#!/usr/bin/env python \n# -*- coding: utf-8 -*-\n\n'
# 	script += 'import os \nimport glob \nimport platform \nimport shutil \n\n'
# 	script += 'cwd = os.getcwd() \ncur_dir_name = os.path.basename(cwd) \npar_dir = os.path.dirname(cwd) \n'
# 	script += "target_dir_name = str(cur_dir_name.split('_')[0]) + '_' + str(int(cur_dir_name.split('_')[1]) - 1) \n"
# 	script += "target = os.path.join(par_dir, target_dir_name, 'Eq*out.udf') \n"
# 	script += "target_list = glob.glob(target) \n\n"
# 	script += "count = 0 \n"
# 	script += "for file_path in target_list: \n"
# 	script += "\tnew_file_name = os.path.basename(file_path).replace('out', 'old_out') \n"
# 	script += "\tdist = os.path.join(cwd, new_file_name) \n"
# 	script += "\tprint('Copying from ', os.path.join(target_dir_name, os.path.basename(file_path))) \n"
# 	script += "\tprint('to ', os.path.join(cur_dir_name, os.path.basename(os.path.dirname(file_path)), new_file_name)) \n\tprint() \n"
# 	script += "\tshutil.copy2(file_path, dist) \n"
# 	script += "\tcount += 1 \n"
# 	script += "print('Copied', count, 'files. Finished !') \n"
# 	return script





################################################################################
#      Main     #
################################################################################
if __name__=='__main__':
	main()
