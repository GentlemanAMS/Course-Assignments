#include "ooo_cpu.h"
#include <map>
#include <algorithm>
#include <array>
#include <bitset>
#include <deque>

/*
Define bimodal parameters
*/
constexpr std::size_t BIMODAL_TABLE_SIZE = 16384;
constexpr std::size_t BIMODAL_PRIME = 16381;
constexpr std::size_t COUNTER_BITS = 2;

/*
Define gshare parameters
*/
#define GLOBAL_HISTORY_LENGTH 14
#define GLOBAL_HISTORY_MASK (1 << GLOBAL_HISTORY_LENGTH) - 1
#define GS_HISTORY_TABLE_SIZE 16384

/*
Define perceptron parameters
*/
constexpr std::size_t PERCEPTRON_HISTORY = 24; // history length for the global history shift register
constexpr std::size_t PERCEPTRON_BITS = 8;     // number of bits per weight
constexpr std::size_t NUM_PERCEPTRONS = 163;
constexpr int THETA = 1.93 * PERCEPTRON_HISTORY + 14; // threshold for training
constexpr std::size_t NUM_UPDATE_ENTRIES = 100; // size of buffer for keeping 'perceptron_state' for update

/*
bimodal class
*/
class bimodal
{
public:

  void bm_initialize_branch_predictor()
  {
    
  }

  uint32_t bm_predict(uint64_t ip)
  {
    uint32_t hash = ip % BIMODAL_PRIME;

    return hash;
  }

  uint32_t bm_update(uint64_t ip)
  {
    uint32_t hash = ip % BIMODAL_PRIME;
    return hash;
  }

};

/*
gshare class
*/
class gshare
{
public:

  void gs_initialize_branch_predictor()
  {

  }

  unsigned int gs_table_hash(uint64_t ip, int bh_vector)
  {
    unsigned int hash = ip ^ (ip >> GLOBAL_HISTORY_LENGTH) ^ (ip >> (GLOBAL_HISTORY_LENGTH * 2)) ^ bh_vector;
    hash = hash % GS_HISTORY_TABLE_SIZE;
    return hash;
  }

  uint32_t gs_predict(uint64_t ip, int gs_index)
  {
    int gs_hash = gs_table_hash(ip, gs_index);

    return gs_hash;
    
  }

  int gs_update(uint64_t ip, int gs_index)
  {
    int gs_hash = gs_table_hash(ip, gs_index);
    return gs_hash;
  }
};

/*
class perceptron
*/
template <typename T, std::size_t HISTLEN, std::size_t BITS>
class perceptron
{
  T bias = 0;
  std::array<T, HISTLEN> weights = {};

public:
  // maximum and minimum weight values
  constexpr static T max_weight = (1 << (BITS - 1)) - 1;
  constexpr static T min_weight = -(max_weight + 1);

  
  T perc_predict(std::bitset<HISTLEN> history)
  {
    auto output = bias;

    // find the (rest of the) dot product of the history register and the
    // perceptron weights.
    for (std::size_t i = 0; i < std::size(history); i++) {
      if (history[i])
        output += weights[i];
      else
        output -= weights[i];
    }

    return output;
  }

  void perc_update(bool result, std::bitset<HISTLEN> history)
  {
    // if the branch was taken, increment the bias weight, else decrement it,
    // with saturating arithmetic
    if (result)
      bias = std::min(bias + 1, max_weight);
    else
      bias = std::max(bias - 1, min_weight);

    // for each weight and corresponding bit in the history register...
    auto upd_mask = result ? history : ~history; // if the i'th bit in the history positively
                                                 // correlates with this branch outcome,
    for (std::size_t i = 0; i < std::size(upd_mask); i++) {
      // increment the corresponding weight, else decrement it, with saturating
      // arithmetic
      if (upd_mask[i])
        weights[i] = std::min(weights[i] + 1, max_weight);
      else
        weights[i] = std::max(weights[i] - 1, min_weight);
    }
  }
};

/* 'perceptron_state' - stores the branch prediction and keeps information
 * such as output and history needed for updating the perceptron predictor
 */
struct perceptron_state {
  uint64_t ip = 0;
  bool prediction = false;                     // prediction: 1 for taken, 0 for not taken
  int output = 0;                              // perceptron output
  std::bitset<PERCEPTRON_HISTORY> history = 0; // value of the history register yielding this prediction
};



// Maintain a table of perceptrons
std::map<O3_CPU*, std::array<perceptron<int, PERCEPTRON_HISTORY, PERCEPTRON_BITS>, NUM_PERCEPTRONS>> perceptrons;
// state for updating perceptron predictor
std::map<O3_CPU*, std::deque<perceptron_state>> perceptron_state_buf;
// real global history - updated when the perceptron predictor is updated
std::map<O3_CPU*, std::bitset<PERCEPTRON_HISTORY>> global_history;
// speculative global history - updated by predictor
std::map<O3_CPU*, std::bitset<PERCEPTRON_HISTORY>> spec_global_history; 

// Maintain table for bimodal predictions
std::map<O3_CPU*, std::array<int, BIMODAL_TABLE_SIZE>> bimodal_table;
std::map<O3_CPU*, bimodal> b;

// Maintain a table of values for the gshare predications
std::map<O3_CPU*, int> branch_history_vector;
std::map<O3_CPU*, std::array<int, GS_HISTORY_TABLE_SIZE>> gs_history_table;
std::map<O3_CPU*, int> my_last_prediction;
std::map<O3_CPU*, gshare> g;

// Maintain a table of weights for the three predictors based on cpu call
std::map<O3_CPU*, std::array<float, 4>> pred_weights;

// Maintain a table of the last output and prediction of bimodal and gshare and our predictor
std::map<O3_CPU*, std::array<bool, 3>> pred_buffer;
std::map<O3_CPU*, std::array<uint32_t, 3>> out_buffer;

// out_buffer, pred_buffer -> 0=>bimodal, 1=>gshare, 2=>model



void O3_CPU::initialize_branch_predictor() 
{
  bimodal_table[this] = {};
  branch_history_vector[this] = 0;
  my_last_prediction[this] = 0;
  for (int i = 0; i < GS_HISTORY_TABLE_SIZE; i++)
    gs_history_table[this][i] = 2; // 2 is slightly taken
  pred_weights[this] = {1/3, 1/3, 1/3, 0};
} 


uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  /*
  Predict using perceptron
  */
  // hash the address to get an index into the table of perceptrons
  auto index = ip % NUM_PERCEPTRONS;
  auto perc_output = perceptrons[this][index].perc_predict(spec_global_history[this]);
  bool perc_prediction = (perc_output >= 0);

  spec_global_history[this] <<= 1;
  spec_global_history[this].set(0, perc_prediction);

  // record the various values needed to update the predictor
  perceptron_state_buf[this].push_back({ip, perc_prediction, perc_output, spec_global_history[this]});
  if (std::size(perceptron_state_buf[this]) > NUM_UPDATE_ENTRIES)
    perceptron_state_buf[this].pop_front();

  /*
  Predict using bimodal
  */
  auto bm_hash = b[this].bm_predict(ip);
  auto bm_output = bimodal_table[this][bm_hash] - (1 << (COUNTER_BITS - 1));
  bool bm_prediction = bm_output >= (1 << (COUNTER_BITS - 1));
  out_buffer[this][0] = bm_output;
  pred_buffer[this][0] = bm_prediction;

  /*
  Predict using gshare
  */
  int gs_index = branch_history_vector[this];
  int gs_hash = g[this].gs_predict(ip, gs_index);
  int pred_ = 1;
  if (gs_history_table[this][gs_hash] >= 2)
    pred_ = 1;
  else
    pred_ = 0;

  my_last_prediction[this] = pred_;

  auto gs_output = gs_history_table[this][gs_hash] - 2;
  bool gs_prediction = gs_output >= 2;
  out_buffer[this][1] = gs_output;
  pred_buffer[this][1] = gs_prediction;

  /*
  Combine the three predictions and update weights
  */
  auto model_output = perc_prediction*pred_weights[this][0] + bm_output*pred_weights[this][1] + gs_output*pred_weights[this][2] + pred_weights[this][3];
  bool model_prediction = model_output >= 0;
  out_buffer[this][2] = static_cast<int32_t>(model_output);
  pred_buffer[this][2] = model_prediction;

  // Make final prediction based on the weighted output of the three predictors  
  return model_prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(perceptron_state_buf[this]), std::end(perceptron_state_buf[this]), [ip](auto x) { return x.ip == ip; });
  if (state == std::end(perceptron_state_buf[this]))
    return; // Skip update because state was lost

  auto [_ip, perc_prediction, perc_output, history] = *state;
  perceptron_state_buf[this].erase(state);

  // update the real global history shift register
  global_history[this] <<= 1;
  global_history[this].set(0, taken);

  // if this branch was mispredicted, restore the speculative history to the
  // last known real history
  if (perc_prediction != taken){
    spec_global_history[this] = global_history[this];
  }

  // If the model output does not match with the taken output, update the weights
  if (pred_buffer[this][2] != taken){
    // Increase weights for which prediction matches with taken, decrease for the others
    
    // Weight associated with bimodal
    // if (pred_buffer[this][0] == taken){
    //   pred_weights[this][1] += (out_buffer[this][0]-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }
    // else{
    //   pred_weights[this][1] -= (out_buffer[this][0]-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }

    if (pred_buffer[this][0] == taken){
      pred_weights[this][1] += (out_buffer[this][0]-out_buffer[this][2]);
    }
    else{
      pred_weights[this][1] -= (out_buffer[this][0]-out_buffer[this][2]);
    }

    // Weight associated with gshare
    // if (pred_buffer[this][1] == taken){
    //   pred_weights[this][2] += (out_buffer[this][1]-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }
    // else{
    //   pred_weights[this][2] -= (out_buffer[this][1]-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }

    if (pred_buffer[this][1] == taken){
      pred_weights[this][2] += (out_buffer[this][1]-out_buffer[this][2]);
    }
    else{
      pred_weights[this][2] -= (out_buffer[this][1]-out_buffer[this][2]);
    }

    // Weight associated with gshare
    // if (perc_prediction == taken){
    //   pred_weights[this][0] += (perc_output-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }
    // else{
    //   pred_weights[this][0] -= (perc_output-out_buffer[this][2])/(pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    // }

    if (perc_prediction == taken){
      pred_weights[this][0] += (perc_output-out_buffer[this][2]);
    }
    else{
      pred_weights[this][0] -= (perc_output-out_buffer[this][2]);
    }
  }

  else{
    if (pred_buffer[this][2] == true){
      pred_weights[this][3] += (pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    }
    else{
      pred_weights[this][3] -= (pred_weights[this][0]+pred_weights[this][1]+pred_weights[this][2]+pred_weights[this][3]);
    }
  }

  // update perceptron model
  auto index = ip % NUM_PERCEPTRONS;
  // if the output of the perceptron predictor is outside of the range
  // [-THETA,THETA] *and* the prediction was correct, then we don't need to
  // adjust the weights
  if ((perc_output <= THETA && perc_output >= -THETA) || (perc_prediction != taken))
    perceptrons[this][index].perc_update(taken, history);

  // Update bimodal 
  uint32_t bm_hash = b[this].bm_update(ip);
  if (taken)
    bimodal_table[this][bm_hash] = std::min(bimodal_table[this][bm_hash] + 1, (1 << COUNTER_BITS) - 1);
  else
    bimodal_table[this][bm_hash] = std::max(bimodal_table[this][bm_hash] - 1, 0);

  // Update gshare
  int gs_hash = g[this].gs_update(ip, branch_history_vector[this]);
  if (taken == 1) {
    if (gs_history_table[this][gs_hash] < 3)
      gs_history_table[this][gs_hash]++;
  } else {
    if (gs_history_table[this][gs_hash] > 0)
      gs_history_table[this][gs_hash]--;
  }

  // update branch history vector
  branch_history_vector[this] <<= 1;
  branch_history_vector[this] &= GLOBAL_HISTORY_MASK;
  branch_history_vector[this] |= taken;
}


